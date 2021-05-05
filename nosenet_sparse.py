import nosenetfunctions as nosenetF
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class PositiveLinear(nn.Module):
	""" 
	Linear layer with positive weights constraint.
	Parameters:
		in_features: input size
		out_features: output size
		sparse: whether to use a sparse multiplication, input must be a sparse coo tensor (default=False)

	Bias is not used when sparse is True.
	"""
	__constants__ = ['in_features', 'out_features', 'sparse']
	in_features: int
	out_features: int
	sparse: bool
	weight: torch.Tensor
	def __init__(self, in_features, out_features, sparse=False):
		super(PositiveLinear, self).__init__()
		self.in_features = in_features
		self.out_features = out_features
		self.sparse = sparse
		self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
		if sparse: # no bias when sparse
			self.register_parameter('bias', None)
		else:
			self.bias = nn.Parameter(torch.Tensor(out_features))
			
		self.reset_parameters()

	def reset_parameters(self):
		# weights
		nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
		# bias
		if self.bias is not None:
			fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
			bound = 1 / math.sqrt(fan_in)
			nn.init.uniform_(self.bias, -bound, bound)

	def forward(self, input):
		self.weight.data = self.weight.data.clamp(min=0)
		if self.sparse:
			return torch.sparse.mm(input, self.weight.t())
		return F.linear(input, self.weight, self.bias)
	
	def extra_repr(self):
		return 'in_features={}, out_features={}, bias={}, sparse={}'.format(
			self.in_features, self.out_features, self.bias is not None, self.sparse
		)


class WTA(nn.Module):
	"""
	implement winner take all with pytorch
	
	parameters: 
	k: number of top values to return
	dim: dimension along which to perform the WTA

	return a pytorch sparse coo tensor
	"""
	__constants__ = ['k','dim']
	k: int
	dim: int
	def __init__(self, k, dim=1):
		super(WTA, self).__init__()
		self.k = k
		self.dim = dim

	def forward(self, x):
		(valuesM, indicesM) = torch.topk(x, self.k, dim=self.dim)
		nb_values = indicesM.shape[0]*indicesM.shape[1]
		values = np.zeros((1,nb_values))
		indices = np.zeros((2,nb_values))
		row_len = indicesM.shape[1]
		for row_idx in range(indicesM.shape[0]):
			start = row_idx*row_len 
			end = start + row_len
			indicesM = indicesM.cpu()
			valuesM = valuesM.cpu()
			indices[1, start:end] = indicesM[row_idx,:]
			indices[0, start:end] = np.repeat(row_idx,row_len)
			values[0, start:end] = valuesM[row_idx,:]
		return torch.sparse_coo_tensor(indices, values[0,:], size=x.shape, dtype=torch.float32).to(x.device)

	def extra_repr(self):
		return 'k={}, dim={}'.format(self.k, self.dim)


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
		self.WTA = WTA(k=self.hash_length)
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
		x = torch.sparse.mm(self.weight, input.t()).t().to(input.device)
		x = self.WTA(x)
		return x

		#coo = self.OM.MB_sparsify(x)
		#tvalues = coo.data
		#tindices = np.vstack((coo.row, coo.col))
		#print(coo.shape)
		#print(tvalues.shape)
		#print(tindices.shape)
		#print(x.shape)
		#print(indices.shape, indices)
		#print(values.shape, values)
		#return torch.sparse_coo_tensor(indices, values[0,:], size=x.shape, dtype=torch.float32)

	
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
		nb_classes = params['NB_CLASSES']
		self.sparse_hebbian = params['sparse_hebbian']
		self.fc1 = MB_projection(params)
		#self.fc2 = WTA(params['HASH_LENGTH'])
		self.fc2 = PositiveLinear(nb_features, nb_classes, sparse=self.sparse_hebbian)

	def forward(self, x):
		x = self.fc1(x)
		#print(x)
		x = self.fc2(x)
		#print(x)
		#x = torch.sigmoid(x)
		return x

class NoseNetDeep(nn.Module):
	"""
	Nosenet neural network
	"""
	def __init__(self, params):
		super(NoseNetDeep, self).__init__()
		nb_features = params['NB_FEATURES'] * params['DIM_EXPLOSION_FACTOR']
		nb_classes = params['NB_CLASSES']
		self.sparse_hebbian = params['sparse_hebbian']
		self.projection = MB_projection(params)
		#self.fc2 = WTA(params['HASH_LENGTH'])
		reduction1 = max((nb_features//100, nb_classes))
		reduction2 = max((reduction1//2, nb_classes))
		self.hebbian = PositiveLinear(nb_features, reduction1, sparse=self.sparse_hebbian)
		self.fc3 = nn.Linear(reduction1, reduction2)
		self.fc4 = nn.Linear(reduction2, nb_classes)
		self.dropout = nn.Dropout(0.2)
		#self.softmax = nn.Softmax(dim=1)

	def forward(self, x):
		x = self.projection(x)
		#print(x)
		x = F.relu(self.hebbian(x))
		x = F.relu(self.fc3(x))
		x = self.dropout(x)
		x = self.fc4(x)
		#x = self.softmax(x)
		#print(x)
		x = torch.sigmoid(x)
		return x
