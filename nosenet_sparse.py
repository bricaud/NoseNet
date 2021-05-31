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

	def forward(self, x):
		self.weight.data = self.weight.data.clamp(min=0)
		if self.sparse:
			return torch.sparse.mm(x, self.weight.t())
		return F.linear(x, self.weight, self.bias)
	
	def extra_repr(self):
		return 'in_features={}, out_features={}, bias={}, sparse={}'.format(
			self.in_features, self.out_features, self.bias is not None, self.sparse
		)


class WTA(nn.Module):
	"""
	implement winner take all with pytorch
	
	parameters: 
	active_outputs: number of top values to return
	dim: dimension along which to perform the WTA

	return a pytorch sparse coo tensor
	"""
	__constants__ = ['active_outputs','dim']
	active_outputs: int
	dim: int
	def __init__(self, active_outputs, dim=1):
		super(WTA, self).__init__()
		self.active_outputs = active_outputs
		self.dim = dim

	def forward(self, x):
		(valuesM, indicesM) = torch.topk(x, self.active_outputs, dim=self.dim)
		nb_values = indicesM.shape[0] * indicesM.shape[1]
		values = np.zeros((1,nb_values))
		indices = np.zeros((2,nb_values))
		row_len = indicesM.shape[1]
		indicesM = indicesM.cpu()
		valuesM = valuesM.cpu()		
		for row_idx in range(indicesM.shape[0]):
			start = row_idx*row_len 
			end = start + row_len
			indices[1, start:end] = indicesM[row_idx,:]
			indices[0, start:end] = np.repeat(row_idx,row_len)
			values[0, start:end] = valuesM[row_idx,:]
		return torch.sparse_coo_tensor(indices, values[0,:], size=x.shape, dtype=torch.float32).to(x.device)

	def extra_repr(self):
		return 'active_outputs={}, dim={}'.format(self.active_outputs, self.dim)


class AL_projection(nn.Module):
	""" 
	MB projection Neural net
	Perform the projection into the mushroom body.
	No learning is involved in this layer.
	"""
	__constants__ = ['in_features', 'out_features']
	in_features: int
	out_features: int
	weight: torch.Tensor
	def __init__(self, in_features, out_features):
		super(AL_projection, self).__init__()
		self.in_features = in_features
		self.out_features = out_features
		self.ALweight = nn.Parameter(torch.empty(self.out_features, self.in_features), requires_grad=False)
		self.reset_parameters()

	def reset_parameters(self):
		AL = nosenetF.proj_matrix(self.out_features, self.in_features, 'DG')
		#print(np.isnan(np.sum(AL.numpy())))
		self.ALweight.data = torch.from_numpy(AL).type(torch.FloatTensor)

	def forward(self, x):
		x = torch.matmul(self.ALweight, x.t()).t()
		return x

	def extra_repr(self):
		return 'in_features={}, out_features={}'.format(self.in_features, self.out_features)



class MB_projection(nn.Module):
	""" 
	MB projection Neural net
	Perform the projection into the mushroom body.
	No learning is involved in this layer.
	"""
	__constants__ = ['in_features', 'out_features', 'projection_type', 'nb_proj_entries']
	in_features: int
	out_features: int
	projection_type: str
	nb_proj_entries: int
	weight: torch.Tensor
	def __init__(self, in_features, dim_explosion, projection_type, nb_proj_entries):
		super(MB_projection, self).__init__()
		self.in_features = in_features	
		self.dim_explosion = dim_explosion
		self.out_features = self.in_features * self.dim_explosion
		self.projection_type = projection_type
		self.nb_proj_entries = nb_proj_entries
		self.MBweight = nn.Parameter(torch.sparse_coo_tensor(size=(self.out_features, self.in_features)),
									 requires_grad=False)
		self.reset_parameters()

	def reset_parameters(self):
		MB = nosenetF.proj_matrix(self.out_features, self.in_features, self.projection_type, self.nb_proj_entries)
		values = MB.data
		indices = np.vstack((MB.row, MB.col))
		self.MBweight.data = torch.sparse_coo_tensor(indices, values, (self.out_features, self.in_features), dtype=torch.float32)

	def forward(self, x):
		x = torch.sparse.mm(self.MBweight, x.t()).t().to(x.device)
		return x

	def extra_repr(self):
		return 'in_features={}, out_features={}, projection_type={}, nb_proj_entries={}'.format(
			self.in_features, self.out_features, self.projection_type, self.nb_proj_entries
		)

class NoseNet(nn.Module):
	"""
	Nosenet neural network
	"""
	def __init__(self, params):
		super(NoseNet, self).__init__()
		self.in_features = params['NB_FEATURES']
		nb_classes = params['NB_CLASSES']
		self.sparse_hebbian = params['sparse_hebbian']
		self.AL_requested = params['AL_projection']

		# AL projection
		if self.AL_requested:
			# create AL
			self.AL_input_size = self.in_features
			nb_PNs = int(self.in_features * params['PNS_REDUCTION_FACTOR'])
			self.AL_output_size = nb_PNs
			self.MB_input_size = self.AL_output_size
			self.AL_projection = AL_projection(self.AL_input_size, self.AL_output_size)
		else:
			self.MB_input_size = self.in_features
		self.out_features = self.MB_input_size * params['DIM_EXPLOSION_FACTOR']
		self.MB_projection = MB_projection(self.MB_input_size, params['DIM_EXPLOSION_FACTOR'],
											params['PROJECTION_TYPE'], params['NB_PROJ_ENTRIES'])
		self.hash_length = int(params['MB_ACTIVITY_RATIO'] * self.out_features)
		self.WTA = WTA(active_outputs=self.hash_length, dim=1)
		self.hebbian = PositiveLinear(self.out_features, nb_classes, sparse=self.sparse_hebbian)

		#self.hebbian = PositiveLinear(self.MB_input_size, nb_classes, sparse=self.sparse_hebbian)

	def forward(self, x):
		#x = x**(1/10)
		if self.AL_requested:
			#print('input',x.shape)
			#print(np.isnan(np.sum(x.numpy())))
			x = self.AL_projection(x)
			# nonlinearity
			x = torch.sigmoid(x)
		x = self.MB_projection(x)
		x = self.WTA(x)
		x = self.hebbian(x)
		#x = torch.sigmoid(x)
		return x

class NoseNetDeep(nn.Module):
	"""
	Nosenet neural network
	"""
	def __init__(self, params):
		super(NoseNetDeep, self).__init__()
		nb_classes = params['NB_CLASSES']
		reduction1 = params['feedforward_layers'][0]
		reduction2 = params['feedforward_layers'][1]
		###
		nosenet_requested = True #for testing without nosenet
		if nosenet_requested:
			nosenet_params = params.copy()
			nosenet_params['NB_CLASSES'] = reduction1
			self.nosenet = NoseNet(nosenet_params)
		else:
			self.nosenet = nn.Linear(params['NB_FEATURES'], reduction1)
		###
		self.fcx1 = nn.Linear(reduction1, reduction2)
		self.dropout = nn.Dropout(params['dropout'])		
		self.fcx2 = nn.Linear(reduction2, nb_classes)
		#self.softmax = nn.Softmax(dim=1)

	def forward(self, x):
		#x = self.projection(x)
		#print(x)
		x = F.relu6(self.nosenet(x)*6)/6
		x = F.relu(self.fcx1(x))
		x = self.dropout(x)
		x = self.fcx2(x)
		#x = self.softmax(x)
		#print(x)
		#x = torch.sigmoid(x)
		return x
