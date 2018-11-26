import torch, random, pdb
import numpy as np
from torch import nn
from torch.nn import functional as F
from dnc import DNC
from collections import deque
from itertools import chain


class CMDrop(nn.Module):
	"""Modified from original codes by Yarin Gal."""
	def __init__(self, *layers):
		super(CMDrop, self).__init__()
		self.layers = nn.Sequential(*layers)

	def forward(self, incoming, logits=None, feature_axis=1, regulariser=[], sparsity=[0.]):
		if logits is None:
			return self.layers(incoming)
		else:
			ξ, reg = self.ccd(incoming, logits, feature_axis)
			regulariser.append(reg)
			sparsity[0] = sparsity[0] + ξ.sum()
			return self.layers(ξ * incoming)

	def ccd(self, incoming, logits, feature_axis):
		eps = 1e-7
		t = 0.1
		if self.training:
			pr = F.sigmoid(logits)
			us = torch.rand(incoming.size(), device=incoming.device)
			cdp = ((pr + eps).log() - (1 - pr + eps).log()
				  +(us + eps).log() - (1 - us + eps).log())
			cdp = F.sigmoid(cdp / t)

			reg = self.regularisation(incoming.size(0), incoming.size(feature_axis), pr)
			
			return (1 - cdp), reg  # retain prob, regulariser
		else:
			return (1 - F.sigmoid(logits)), torch.zeros(1, device=incoming.device)

	def sum_l2_squared(self):
		l2sq = 0
		for p in self.layers.parameters():
			l2sq += p.pow(2).sum()
		return l2sq
	
	def regularisation(self, bs, dim, drop_p, length_scale=1e-4):
		eps = 1e-7
		weights_regulariser, dropout_regulariser = 0, 0
		weights_regulariser += length_scale ** 2 * self.sum_l2_squared() / bs
		dropout_regulariser += drop_p * torch.log(drop_p + eps)
		dropout_regulariser += (1. - drop_p) * torch.log(1. - drop_p + eps)
		dropout_regulariser *= 2 * bs * dim
		regulariser = weights_regulariser + dropout_regulariser
		return regulariser


class ConflictMonitoringNet(nn.Module):
	def __init__(self, gpu_id=0, 
				 input_size=1, 
				 output_size=1, 
				 num_layers=4, 
				 hidden_size=128, 
				 rnn_type='gru', 
				 rnn_num_layers=2, 
				 rnn_hidden_size=128):
		super(ConflictMonitoringNet, self).__init__()
		self.gpu_id = gpu_id
		self.input_size = input_size
		self.output_size = output_size
		self.num_layers = num_layers
		self.hidden_size = hidden_size
		self.rnn_type = rnn_type
		self.rnn_num_layers = rnn_num_layers
		self.rnn_hidden_size = rnn_hidden_size
		self.cmd_regulariser = 0
		self.layers = nn.Sequential(
			CMDrop(nn.Conv1d(  input_size, hidden_size, 1), nn.ReLU()),
			CMDrop(nn.Conv1d( hidden_size, hidden_size, 1), nn.ReLU()),
			CMDrop(nn.Conv1d( hidden_size, hidden_size, 1), nn.ReLU()),
			CMDrop(nn.Conv1d( hidden_size, output_size, 1))
			)

		if self.rnn_type =='gru':
			self.rnn = nn.GRU(
				input_size=output_size,
				hidden_size=rnn_hidden_size,
				num_layers=rnn_num_layers,
				batch_first=True)
			self.rnn_hidden = None
		elif self.rnn_type == 'dnc':
			self.rnn = DNC(
				input_size=output_size,
				hidden_size=rnn_hidden_size,
				rnn_type='gru',
				num_layers=rnn_num_layers,
				nr_cells=10,
				cell_size=64,
				read_heads=4,
				batch_first=True,
				gpu_id=self.gpu_id)
			self.rnn_hidden = (controller_hidden, memory, read_vectors) = (None, None, None)

		self.rnn_output_layer = nn.Conv1d(rnn_hidden_size, num_layers, 1)

	def reset_rnn(self):
		self.cmd_regulariser = 0
		if self.rnn_type == 'gru':
			self.rnn_hidden = None
		elif self.rnn_type == 'dnc':
			self.rnn_hidden = (controller_hidden, memory, read_vectors) = (None, None, None)

	def forward(self, x, y, carryover=1):
		# x, y : [num_batch, num_t, input_size]
		y_hat_1, loss_0 = self.first_pass(x, y)
		y_hat_2, loss_1 = self.second_pass(x, y, loss_0, carryover=carryover)

		if carryover > 0:
			y_hat = torch.cat([y_hat_1[:, :carryover], y_hat_2], dim=1)
		else:
			y_hat = y_hat_2

		return y_hat, loss_0, loss_1

	def first_pass(self, x, y, logits_split=None):
		h = x.permute(0, 2, 1)
		if logits_split is None:
			y_hat = self.layers(h).permute(0, 2, 1)
		else:
			reg = []
			for i, module in enumerate(self.layers):
				h = module(h, logits=logits_split[i], regulariser=reg)
			self.cmd_regulariser = sum([r.sum() for r in reg]) / len(reg)
			y_hat = h.permute(0, 2, 1)

		return y_hat, (y_hat - y).pow(2)
		
	def second_pass(self, x, y, l0, carryover):
		logits = self.shifter_logits(l0.detach())
		y_hat, l1 = self.first_pass(*self.on_carryover(x, y, logits, carryover))

		return y_hat, l1

	def shifter_logits(self, loss):
		if self.rnn_type == 'gru':
			logits, h_n = self.rnn(loss, self.rnn_hidden)
			self.rnn_hidden = h_n
		elif self.rnn_type == 'dnc':
			logits, (controller_hidden, memory, read_vectors) = self.rnn(loss, self.rnn_hidden)
			self.rnn_hidden = (controller_hidden, memory, read_vectors)
		logits = self.shifter_output(logits)

		return logits

	def shifter_output(self, out):
		return self.rnn_output_layer(out.permute(0, 2, 1)).split(1, dim=1)

	def on_carryover(self, x, y, logits, carry):
		if carry > 0:
			x = x[:, carry:]
			y = y[:, carry:]
			logits = [lgt[:, :, :-carry] for lgt in logits]
		return x, y, logits
	
	def predict(self, x, y):
		sum_loss = 0.
		y_hat = []
		N = 0
		for i, (u, v) in enumerate(zip(x.split(1, dim=1), y.split(1, dim=1))):
			if i == 0:
				yh, carried_loss = self.first_pass(u, v)
			else:
				yh, carried_loss = self.second_pass(u, v, carried_loss, carryover=0)
			y_hat.append(yh)
			sum_loss += carried_loss
			N += 1
		return torch.cat(y_hat, dim=1), sum_loss / N

	def predict1(self, x, c):
		logits = self.shifter_logits(c)


class UnitWiseConflictMonitoringNet(ConflictMonitoringNet):
	def __init__(self, *args, **kwargs):
		super(UnitWiseConflictMonitoringNet, self).__init__(*args, **kwargs)
		self.rnn_output_layer = nn.ModuleList([
				nn.Sequential(
					nn.Conv1d(self.rnn_hidden_size, 256, 1), nn.ReLU(),
					nn.Conv1d(                 256, self.input_size, 1)),
				nn.Sequential(
					nn.Conv1d(self.rnn_hidden_size, 256, 1), nn.ReLU(),
					nn.Conv1d(                 256, self.hidden_size, 1)),
				nn.Sequential(
					nn.Conv1d(self.rnn_hidden_size, 256, 1), nn.ReLU(),
					nn.Conv1d(                 256, self.hidden_size, 1)),
				nn.Sequential(
					nn.Conv1d(self.rnn_hidden_size, 256, 1), nn.ReLU(),
					nn.Conv1d(                 256, self.hidden_size, 1)),
			])

	def forward(self, *args, **kwargs):
		return super().forward(*args, **kwargs)

	def shifter_output(self, out):
		ret = [layer(out.permute(0, 2, 1)) for layer in self.rnn_output_layer]
		return ret


# -----------------------------------------------------------------------------
# Models for comparison
class CMBaselineForward(nn.Module):
	def __init__(self, input_size=1, 
				 output_size=1, hidden_size=128):
		super(CMBaselineForward, self).__init__()
		self.layers = nn.Sequential(
			nn.Conv1d( input_size, hidden_size, 1), nn.ReLU(),
			nn.Conv1d(hidden_size, hidden_size, 1), nn.ReLU(),
			nn.Conv1d(hidden_size, hidden_size, 1), nn.ReLU(),
			nn.Conv1d(hidden_size, output_size, 1))
		
	def forward(self, x, y):
		y_hat = self.layers(x.permute(0, 2, 1)).permute(0, 2, 1)
		return y_hat, (y - y_hat).pow(2).sum(-1)


class CMBaselineRecurrent(nn.Module):
	def __init__(self, input_size=1, output_size=1, rnn_num_layers=2, rnn_hidden_size=256):
		super(CMBaselineRecurrent, self).__init__()
		self.rnn = nn.GRU(
				input_size=input_size,
				hidden_size=rnn_hidden_size,
				num_layers=rnn_num_layers,
				batch_first=True)
		self.rnn_out = nn.Conv1d(rnn_hidden_size, output_size, 1)
		
	def forward(self, x, y):
		y_hat = self.rnn_out(self.rnn(x)[0].permute(0, 2, 1)).permute(0, 2, 1)
		
		return y_hat, (y - y_hat).pow(2).sum(-1)
