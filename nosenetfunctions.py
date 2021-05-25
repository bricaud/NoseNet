"""
Olfactory Model
"""

# Authors: Volodymyr MIZ and Benjamin Ricaud
# Open source under the MIT license
# Some parts of the code and the algorithm are taken from https://science.sciencemag.org/content/358/6364/793/tab-figures-data
# v0.1.2

import numpy as np
import random
from scipy.sparse import csr_matrix, coo_matrix, lil_matrix
import scipy

class OlfactoryModel():
    """
    Olfactory model implementation.
    
    Parameters
    ----------
    params: dict, must contain the following parameters:

    'NB_FEATURES': int, required parameter
        Dimensionality of input data.
    
    'DIM_EXPLOSION_FACTOR': int, default=10
        Factor of the increasing dimensionality.
        Dimensionality of the projection is MB_input_size * dim_explosion_factor
    
    'PROJECTION_TYPE': string, default="SB12"
        Projection type. Possible options: SB and DG.
        SB stands for Sparse Binary projection.


    'NB_PROJ_ENTRIES': int, default=6
        Number of combinations of input features in the MB, for each Kenyon cell.
        
    'HASH_LENGTH': int, default=32
        Length of hash encoding. Number of non-zero values in hash encoding (output of the MB).
    
    'WTA': string, default='top'
        WTA stands for Winner Takes All.
        Defines which values are going to represent hash.
        Possible options: 'top', 'bottom', 'random', 'all'.
        
    Examples
    --------
    
    >>> from olfaction import OlfactoryModel
    >>> import numpy as np
    >>> data = np.arange(5000) # generate synthetic data
    >>> data.shape = (100,50)
    >>> OM = OlfactoryModel(params) # initialize the model
    >>> H = OM.get_projection(data) # compute projection
    >>> 
    """
    
    def __init__(self, params):
        if params['AL_projection'] == None:
            self.MB_input_size = params['NB_FEATURES']
        else:
            self.AL_input_size = params['NB_FEATURES']
            self.AL_output_size = params['NB_FEATURES']//10
            self.MB_input_size = self.AL_output_size
        self.dim_explosion_factor = params['DIM_EXPLOSION_FACTOR']
        self.projection_type = params['PROJECTION_TYPE']
        self.nb_proj_entries = params['NB_PROJ_ENTRIES']
        self.hash_length = params['HASH_LENGTH']
        self.WTA = params['WTA']
        self.projection_dim = self.MB_input_size * self.dim_explosion_factor
        self.Hebbian_weights = scipy.sparse.lil_matrix((
                        self.MB_input_size * self.dim_explosion_factor, self.MB_input_size))
        self.Hebbian_batch_weight = 0
        self.__MB = self._create_rand_proj_matrix(self.projection_dim, self.MB_input_size, 
            params['PROJECTION_TYPE'], params['NB_PROJ_ENTRIES'])
        self.__AL = self._create_rand_proj_matrix(self.MB_input_size, self.AL_input_size, 'DG')
    
    def AL_projection(self, X):
        """
        Projection from the input (odors) to the Antenna lobe
        This is a random projection with dimension reduction
        """
        P = self.__AL.dot(X.T).T      
        return P

    def MB_projection(self, X):
        """
        Compute projection
        
        Parameters
        ----------
        X: {array-like, sparse matrix} of shape (n_samples, n_features)
            Data matrix of shape (n_samples, n_features)
            
        Returns
        -------
        Projection matrix of shape (n_samples, self.MB_input_size * self.dim_explosion_factor)
        """
        
        NUM_KENYON = X.shape[1] * self.dim_explosion_factor
        N = X.shape[0]
        
        #P = np.dot(X,np.transpose(self.__M)) # N x NUM_KENYON
        #P = self.__M.dot(X.T).T
        P = self.__MB.dot(X.T).T

        assert P.shape[0] == N
        assert P.shape[1] ==  NUM_KENYON        
        return P


    def MB_sparsify(self,X):
        """
        Apply a sparsifying function to the output of the MB projection
        returns
        -------
        A sparse matrix (scipy.csr_matrix)
        """

        NUM_KENYON = X.shape[1]
        N = X.shape[0]
        # Apply WTA to KCs: firing rates at indices corresponding to top/bot/rand/all KCs; 0s elsewhere.
        if self.WTA == "random":# fix indices for all odors, otherwise, can't compare.
            rand_indices = random.sample(range(NUM_KENYON),self.hash_length)

        col_ind_array = []
        row_ind_array = []
        data_array = []
        
        for i in range(N):

        # Take all neurons.
            if self.WTA == "all":
                assert self.hash_length == NUM_KENYON
                indices = range(NUM_KENYON)

            # Highest firing neurons.
            elif self.WTA == "top":      
                indices = np.argpartition(X[i,:],-self.hash_length)[-self.hash_length:]
                col_ind_array.extend(indices)
                row_ind_array.extend([i] * len(indices))
                data_array.extend(X[i,:][indices])
                
            # Lowest firing neurons.
            elif self.WTA == "bottom": 
                indices = np.argpartition(X[i,:],self.hash_length)[:self.hash_length]  

            # Random neurons. 
            elif self.WTA == "random": 
                indices = rand_indices #random.sample(range(NUM_KENYON),HASH_LENGTH)

            # adaptive
            elif self.WTA == "adaptive":
                pass
            else: assert False
    
        return coo_matrix((data_array, (row_ind_array, col_ind_array)), shape=(N, NUM_KENYON))#.tocsr()
        
    def create_rand_proj_matrix(self, projection_module):
        if projection_module == 'AL':
            nb_cols = self.AL_input_size
            nb_rows = self.MB_input_size
            projection_type = 'DG'  # gaussian random projection
            nb_proj_entries = None
        elif projection_module == 'MB':
            nb_cols = self.MB_input_size
            nb_rows = self.projection_dim
            projection_type = self.projection_type
            nb_proj_entries = self.nb_proj_entries
        else: 
            raise ValueError('Wrong projection module name: {}'.format(projection_module))
        return self._create_rand_proj_matrix(nb_rows, nb_cols, projection_type, nb_proj_entries)


    def _create_rand_proj_matrix(self, nb_rows, nb_cols, projection_type, nb_proj_entries=None):
        """ 
        Creates a random projection matrix of size NUM_KENYON by NUM_PNS. 
        """
    
        if projection_type != 'DG': # DG does not need nb_proj_entries
            num_sample = nb_proj_entries
            assert num_sample <= nb_cols
            ### TODO: in some configurations M is not a matrix with booleans 
            M = lil_matrix((nb_rows,nb_cols), dtype=bool)

            
        # Create a sparse, binary random projection matrix.
        if projection_type == "SB":
            for row in range(nb_rows):
                # Sample NUM_SAMPLE random indices, set these to 1.
                for idx in random.sample(range(nb_cols),num_sample):
                    M[row,idx] = 1
            # Make sure I didn't screw anything up!
            #assert sum(M[row,:]) == num_sample  

        # Create a sparse, binary random projection matrix.
        # a feature can be chosen multiple times (choice with replacement)
        elif projection_type =="SS":
            for row in range(nb_rows):
                for idx in random.choices(range(nb_cols),k=num_sample):
                    M[row,idx] = 1


        # Create a sparse, binary random projection matrix.
        # a feature can be chosen multiple times + k is the mean value of connections
        elif projection_type == "RK":
            for trial in range(nb_rows*num_sample):
                row = random.randrange(nb_rows)
                col = random.randrange(nb_cols)
                M[row,col] += 1


        # Create a sparse, binary random projection matrix, with positive and negative values.
        # a feature can be chosen multiple times + k is the mean value of connections
        elif projection_type == "RL":
            for trial in range(nb_rows*num_sample):
                row = random.randrange(nb_rows)
                col = random.randrange(nb_cols)
                M[row,col] += random.choice([-1,1])

        # Matrix entries are positive numbers between 0 and 1
        elif projection_type == "SR": 
            for row in range(nb_rows):
                for idx in random.sample(range(nb_cols),num_sample):
                    M[row,idx] = 0.1 * np.random.randn() +1 #1

     
        # Matrix entries are binary, 
        # random number of non-zeros (fluctuating around num_sample)
        elif projection_type == "SX":
            delta = 4
            min_val = max([num_sample - delta,1])
            max_val = min([num_sample + delta, nb_cols])          
            for row in range(nb_rows):
                num_sample = np.random.randint(min_val, max_val + 1)
                for idx in random.sample(range(nb_cols),num_sample):
                    M[row,idx] = 1 #np.random.randn() #1
        
        # Matrix entries are positive integers between 0 and 1,
        # random number of non-zeros (fluctuating around num_sample)
        elif projection_type == "SZ":
            delta = 4
            min_val = max([num_sample - delta,1])
            max_val = min([num_sample + delta, nb_cols])            
            for row in range(nb_rows):
                num_sample = np.random.randint(min_val, max_val + 1)
                for idx in random.sample(range(nb_cols),num_sample):
                    M[row,idx] = np.random.randint(1, num_sample + 1)

        # Create a dense, Gaussian random projection matrix.
        elif projection_type == "DG":
            M = np.random.randn(nb_rows, nb_cols)
            return M # M is a dense matrix

        else:
            ValueError('Wrong projection type: {}'.format(projection_type))

        return M.tocoo() #M.tocsr()
    
    def infer_data(self, H):
        """
        Infer data given its hashed version
        
        Parameters
        ----------
        H: {array-like, sparse matrix} of shape (n_samples, n_features)
            Data matrix of shape (n_samples, n_features)
            
        Returns
        -------
        Data matrix of shape (n_samples, nb_cols * self.dim_explosion_factor)
        X: {dense matrix}
        """

        X, residuals, rank, s = np.linalg.lstsq(self.__M, H.T, rcond=None)

        return X.T

    def hebbian(self, MB_proj,train_labels):
        """
        Hebbian Learning process
        train_labels are one-hot-encoded. Multiclasses are allowed.
        train_labels shape is samples x classes
        """
        
        # label normalization matrix
        sum_labels = np.sum(train_labels, axis=0)
        non_zeros = (sum_labels != 0) 
        inv_sum_labels = sum_labels[non_zeros] = 1. / sum_labels[non_zeros]

        # Getting the most representative features
        sum_activity = MB_proj.dot(train_labels).dot(np.diag(inv_sum_labels))
        self.Hebbian_weights = sum_activity
        return

    def infer(self,formula_matrix):
        """
        return the classes of the fomulas in the formula_matrix
        formula_matrix has shape (ingredients x formulas)
        return a softmax classification matrix (formulas x classes)
        """
        pass