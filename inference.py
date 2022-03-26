import numpy as np
import torch
import tqdm
import argparse
from torchvision.utils import save_image
from trainer import origin_no_bit_model, mnist_data_loader, origin_bit_model
from data import fashion_mnist_data_loader, cifar10_data_loader
# model
# model_path = '2dim_nobit_model.pth.tar' # 
model_path = 'cifar10_model.pth.tar'
# model_path = '2dim_20bit_model.pth.tar'
# model_path = '2dim_30bit_model.pth.tar'

# model = origin_no_bit_model()(2, 1)   # nobit
model = origin_bit_model()(48, 3, 2)  # 10bit
# model = origin_bit_model()(20, 1, 10) # 20bit
# model = origin_bit_model()(30, 1, 15) # 30bit


model.cuda()
state_dict = torch.load(model_path)['state_dict']
model.load_state_dict(state_dict)

# data
# data_path = './data/'
train_loader, val_loader = cifar10_data_loader(batch_size=16)
x, y = next(iter(train_loader))
x = x.cuda()

# inference
z, recon = model(x)
num = model.bit2num(z)
print(num)
save_image(torch.concat([x, recon], dim=0) , 'test-mnist.png')

