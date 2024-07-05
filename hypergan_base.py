from abc import ABC, abstractmethod
import torch.nn as nn

class HyperGAN_Base(ABC):

    def __init__(self, args):
        self.sample_size = args.s
        self.latent_width = args.z
        self.ngen = 6 
        args.ensemble_size = args.batch_size

    @abstractmethod
    class Generator(object):
        def __init__(self, args):
            raise NotImplementedError

    @abstractmethod
    def eval_f(self, args):
        raise NotImplementedError

    @abstractmethod
    def restore_models(self, args):
        raise NotImplementedError
    
    @abstractmethod
    def save_models(self, args):
        raise NotImplementedError
            

    
