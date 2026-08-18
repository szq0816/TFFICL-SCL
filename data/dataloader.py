import torch
from torch.utils.data import DataLoader, Dataset
import os
import numpy as np
from torch.fft import fft

class Load_Dataset(Dataset):
    def __init__(self, dataset, args):
        super().__init__()
        self.mode = args.mode
        if self.mode == 'finetune':
                num = int(dataset[0].size(0) * args.percent)
                x_data = dataset[0][:num]
                y_data = dataset[1][:num]
        else:
                x_data = dataset[0]
                y_data = dataset[1]


        if isinstance(x_data, np.ndarray):
            self.t_data = torch.from_numpy(x_data).float()
            self.y_data = torch.from_numpy(y_data).long()
        else:
            self.t_data = x_data.float()
            self.y_data = y_data.long()

        '''Frequency domain'''
        self.f_data = fft(self.t_data).abs()


    def __getitem__(self, index):

        return self.t_data[index], self.f_data[index], self.y_data[index]

    def __len__(self):
        return self.t_data.shape[0]


def Data_Loader(path, args, mode):
    if mode == 'train':
        path = '/home/cty02/code/dataset/{}'.format(path)
        data = torch.load(os.path.join(path, 'train.pt'))
        dataset = Load_Dataset(data, args)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    elif mode == 'finetune':
        path = '/home/cty02/code/dataset/{}'.format(path)
        data = torch.load(os.path.join(path, 'val.pt'))
        dataset = Load_Dataset(data, args)
        loader = DataLoader(dataset, batch_size=args.f_batch_size, shuffle=True, drop_last=True)
    else:
        path = '/home/cty02/code/dataset/{}'.format(path)
        data = torch.load(os.path.join(path, 'test.pt'))
        dataset = Load_Dataset(data, args)
        loader = DataLoader(dataset, batch_size=args.f_batch_size, shuffle=False)

    return loader


