import torch

class Config(object):
    def __init__(self):

        self.seq_len = 1024
        self.num_channels = 1

        '''dropout'''
        self.dropout = 0.1
        '''num_classes'''
        self.num_class = 13

        self.num_prediction = 1
        self.ndim = 512

        self.lamda1=1.0
        self.lamda2=1.0
        self.batch_size=128

        self.tempture = 0.5
        self.projection = 128
        self.head = 1

        self.device = torch.device("cuda")