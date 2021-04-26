import nosenetfunctions as nosenetF
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class PositiveLinear(nn.Module):
	""" 
	Linear layer with positive weights constraint.

	"""
	__constants__ = ['in_features', 'out_features']
	in_features: int
	out_features: int
	weight: torch.Tensor
	def __init__(self, in_features, out_features):
		super(PositiveLinear, self).__init__()
		self.in_features = in_features
		self.out_features = out_features
		self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
		self.bias = nn.Parameter(torch.Tensor(out_features))
		self.reset_parameters()

	def reset_parameters(self):
		# weights
		nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
		# bias
		fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
		bound = 1 / math.sqrt(fan_in)
		nn.init.uniform_(self.bias, -bound, bound)

	def forward(self, input):
		self.weight.data = self.weight.data.clamp(min=0)
		return F.linear(input, self.weight, self.bias)#.exp())
	
	def extra_repr(self):
		return 'in_features={}, out_features={}, bias={}'.format(
			self.in_features, self.out_features, self.bias is not None
		)

class MB_projection(nn.Module):
	""" 
	MB projection Neural net
	Perform the projection into the mushroom body.
	No learning is involved in this layer.
	"""
	__constants__ = ['in_features', 'out_features', 'nb_proj_entries', 'hash_length']
	in_features: int
	out_features: int
	nb_proj_entries: int
	hesh_length: int
	weight: torch.Tensor
	def __init__(self, params):
		super(MB_projection, self).__init__()
		self.in_features = params['NB_FEATURES']
		self.out_features = params['NB_FEATURES'] * params['DIM_EXPLOSION_FACTOR']
		self.nb_proj_entries = params['NB_PROJ_ENTRIES']
		self.hash_length = params['HASH_LENGTH']
		self.OM = nosenetF.OlfactoryModel(params)
		self.weight = nn.Parameter(torch.sparse_coo_tensor(size=(self.out_features, self.in_features)),
									 requires_grad=False)
		self.reset_parameters()

	def reset_parameters(self):
		coo = self.OM.create_rand_proj_matrix()
		values = coo.data
		indices = np.vstack((coo.row, coo.col))
		#i = torch.LongTensor(indices)
		#v = torch.FloatTensor(values)
		#shape = coo.shape
		#torch.sparse.FloatTensor(i, v, torch.Size(shape)).to_dense()
		self.weight.data = torch.sparse_coo_tensor(indices, values, (self.out_features,self.in_features), dtype=torch.float32)

	def forward(self, input):
		#x = self.OM.MB_projection(input, self.weight)
		#P = M.dot(X.T).T
		
		#x = F.linear(input,self.weight)
		#print(input.shape,self.weight.shape)
		x = torch.sparse.mm(self.weight, input.t()).t()
		#x = x.T
		coo = self.OM.MB_sparsify(x)
		values = coo.data
		indices = np.vstack((coo.row, coo.col))
		#i = torch.LongTensor(indices)
		#v = torch.FloatTensor(values)
		#shape = coo.shape
		#torch.sparse.FloatTensor(i, v, torch.Size(shape)).to_dense()
		return torch.sparse_coo_tensor(indices, values, (coo.shape), dtype=torch.float32)

	
	def extra_repr(self):
		return 'in_features={}, out_features={}, nb_proj_entries={}, hash_length={}'.format(
			self.in_features, self.out_features, self.nb_proj_entries, self.hash_length
		)

class NoseNet(nn.Module):
	"""
	Nosenet neural network
	"""
	def __init__(self, params):
		super(NoseNet, self).__init__()
		nb_features = params['NB_FEATURES'] * params['DIM_EXPLOSION_FACTOR']
		self.fc1 = MB_projection(params)
		self.fc2 = PositiveLinear(nb_features, params['NB_CLASSES'])

	def forward(self, x):
		x = self.fc1(x)
		#print(x)
		x = self.fc2(x)
		#print(x)
		#x = torch.sigmoid(x)
		return x
