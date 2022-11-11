import sys

# import numpy as np
import torch
# from matplotlib import pyplot as plt

import dataset
from network import Network
from predict import Predict
from test import Test
from train import Train


def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


if __name__ == '__main__':
    
    def same_seeds(seed):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    
    
    # fix random seed for reproducibility
    same_seeds(666)
    
    mode = 'train' if (sys.argv.__len__() < 1) else sys.argv[1]
    # mode = 'train'
    batch_size = 32
    
    device = get_device()
    
    if mode == 'train':
        
        opt_dim = 10
        epoch_num = 3
        
        tr_loader = dataset.prep_loader(mode, batch_size=batch_size)
        
        model = Network(output_dim=10).to(get_device())
        
        process = Train(data=tr_loader, model=model, num_epoch=epoch_num, learning_rate=0.0001, device=device)
        process.train()
    
    elif mode == 'test':
        tt_loader = dataset.prep_loader(mod=mode, batch_size=batch_size)
        
        model = torch.load('./model.pth').to(get_device())
        
        process = Test(data=tt_loader, model=model, device=device)
        process.test()
    elif mode == 'recognize':
        model = torch.load('./model.pth', map_location=torch.device(get_device()))
        rc_loader = dataset.prep_loader(mod='recognize', batch_size=1)
    
        predict = Predict(data=rc_loader, model=model, device=get_device())
        process = predict.predict()
    
    elif mode == 'opt_data':
        # loader = dataset.plot_data()
        pass
    else:
        print('No mode selected. ')
