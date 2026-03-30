from typing import Literal
from tap import Tap


class BaseArgs(Tap):
    path: str = 'Data/'     # Input data path
    dataset: str = 'ml-1m'
    epochs: int = 100
    batch_size: int = 256
    num_factors: int = 8    # Embedding size
    num_neg: int = 4
    '''Number of negative instances to pair with a positive instance'''
    lr: float = 0.001       # Learning Rate
    learner: Literal['adagrad', 'adam', 'rmsprop', 'sgd'] = 'adam' # Optimizer
    verbose: int = 1
    out: bool = True        # Whether to save the trained model


class GMFArgs(BaseArgs):
    regs: str = '[0,0]'     # Regularization for each layer


class MLPArgs(BaseArgs):
    regs: str = '[0, 0, 0, 0]'     # Regularization for each layer

    layers: str = '[64, 32, 16, 8]'
    '''Size of each layer. Note that the first layer is the concatenation of
    user and item embeddings. So layers[0]/2 is  the embedding size.'''

    reg_layers: str = '[0, 0, 0, 0]'
    '''Regularization for each MLP layer. reg_layers[0] is the regularization
    for embeddings.'''


class NeuMFArgs(BaseArgs):
    layers: str = '[64, 32, 16, 8]'
    '''Size of each layer. Note that the first layer is the concatenation of
    user and item embeddings. So layers[0]/2 is  the embedding size.'''

    reg_layers: str = '[0, 0, 0, 0]'
    '''Regularization for each MLP layer. reg_layers[0] is the regularization
    for embeddings.'''

    reg_mf: float = 0.0      # Regularization for MF embeddings
    mlp_pretrain: str = ''   # Pretrained model file for MLP part (optional)
    mf_pretrain: str = ''    # Pretrained model file for MF part (optional)
