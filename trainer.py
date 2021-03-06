import numpy as np
import torch
import tqdm
import argparse
from torch.optim import Adamax
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image


# Auto Encoder for Minist
from data import mnist_data_loader, cifar10_data_loader


def origin_bit_model():
    import math
    import torch
    import numpy as np
    import torch.nn as nn
    from torch.nn import Parameter
    import torch.nn.functional as F

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Position Attention Module
    class PAM_Module(nn.Module):
        """ Position attention module"""

        # Ref from SAGAN
        def __init__(self, in_dim):
            super(PAM_Module, self).__init__()
            self.chanel_in = in_dim
            down_scale_dim = max(in_dim // 8, 3)
            self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=down_scale_dim, kernel_size=1)
            self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=down_scale_dim, kernel_size=1)
            self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
            self.gamma = Parameter(torch.zeros(1))

            self.softmax = nn.Softmax(dim=-1)

        def forward(self, x):
            m_batchsize, C, height, width = x.size()
            proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
            proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
            energy = torch.bmm(proj_query, proj_key)
            attention = self.softmax(energy)
            proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)

            out = torch.bmm(proj_value, attention.permute(0, 2, 1))
            out = out.view(m_batchsize, C, height, width)

            out = self.gamma * out
            return out


    # Channel Attention Module
    class CAM_Module(nn.Module):
        """ Channel attention module"""

        def __init__(self, in_dim):
            super(CAM_Module, self).__init__()
            self.chanel_in = in_dim

            self.gamma = Parameter(torch.zeros(1))
            self.softmax = nn.Softmax(dim=-1)

        def forward(self, x):
            m_batchsize, C, height, width = x.size()
            proj_query = x.view(m_batchsize, C, -1)
            proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
            energy = torch.bmm(proj_query, proj_key)
            energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
            attention = self.softmax(energy_new)
            proj_value = x.view(m_batchsize, C, -1)

            out = torch.bmm(attention, proj_value)
            out = out.view(m_batchsize, C, height, width)

            out = self.gamma * out
            return out


    def convert2bit(input_n, B):
        num_ = input_n.long()
        exp_bts = torch.arange(0, B).to(device)
        exp_bts = exp_bts.repeat(input_n.shape + (1,))
        bits = torch.div(num_.unsqueeze(-1), 2 ** exp_bts, rounding_mode='trunc')
        bits = bits % 2
        bits = bits.reshape(bits.shape[0], -1).float()
        return bits


    class Bitflow(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, b_):
            ctx.constant = b_
            scale = 2 ** b_
            out = torch.round(x * scale - 0.5)
            out = convert2bit(out, b_)
            return out

        @staticmethod
        def backward(ctx, grad_output):
            b, _ = grad_output.shape
            grad_num = torch.sum(grad_output.reshape(b, -1, ctx.constant), dim=2) / ctx.constant
            return grad_num, None


    class BitLayer(nn.Module):
        def __init__(self, B):
            super(BitLayer, self).__init__()
            self.B = B

        def forward(self, x):
            out = Bitflow.apply(x, self.B)
            return out

    # ResNet Block
    class BasicBlock(nn.Module):
        expansion = 1
        def __init__(self, in_planes, planes, stride=1):
            super(BasicBlock, self).__init__()
            self.conv1 = nn.Conv2d(
                in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                                stride=1, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(planes)

            self.shortcut = nn.Sequential()
            if stride != 1 or in_planes != self.expansion*planes:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion*planes,
                            kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion*planes)
                )

        def forward(self, x):
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out += self.shortcut(x)
            out = F.relu(out)
            return out

    # ResNet BottleNeck
    class Bottleneck(nn.Module):
        expansion = 4
        def __init__(self, in_planes, planes, stride=1):
            super(Bottleneck, self).__init__()
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                                stride=stride, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(planes)
            self.conv3 = nn.Conv2d(planes, self.expansion *
                                planes, kernel_size=1, bias=False)
            self.bn3 = nn.BatchNorm2d(self.expansion*planes)

            self.shortcut = nn.Sequential()
            if stride != 1 or in_planes != self.expansion*planes:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion*planes,
                            kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion*planes)
                )

        def forward(self, x):
            out = F.relu(self.bn1(self.conv1(x)))
            out = F.relu(self.bn2(self.conv2(out)))
            out = self.bn3(self.conv3(out))
            out += self.shortcut(x)
            out = F.relu(out)
            return out

    # ResNet Backbone As Encoder
    class ResNet(nn.Module):
        def __init__(self, block, num_blocks, input_channel=32, hidden_dim=128, bit_expand=2):
            super(ResNet, self).__init__()
            self.in_planes = 64
            self.input_channel = input_channel

            self.conv1 = nn.Conv2d(input_channel, 64, kernel_size=3,
                                stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
            self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
            self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
            self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
            
            self.feat = nn.Sequential(nn.Linear(512*block.expansion, hidden_dim // bit_expand),
                                    nn.BatchNorm1d(hidden_dim // bit_expand, eps=1e-6))

            self.sigmoid = nn.Sigmoid()
            self.bit_layer = BitLayer(bit_expand)

        def _make_layer(self, block, planes, num_blocks, stride):
            strides = [stride] + [1]*(num_blocks-1)
            layers = []
            for stride in strides:
                layers.append(block(self.in_planes, planes, stride))
                self.in_planes = planes * block.expansion
            return nn.Sequential(*layers)

        def forward(self, x):
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
            out = F.avg_pool2d(out, 4)
            out = out.view(out.size(0), -1)
            out = self.feat(out)
            out = self.sigmoid(out)
            out = self.bit_layer(out)
            return out


    def ResNet18(input_channel=32, hidden_dim=128, bit_expand=2):
        return ResNet(BasicBlock, [2, 2, 2, 2], input_channel, hidden_dim, bit_expand)


    def ResNet34(input_channel=32, hidden_dim=128, bit_expand=2):
        return ResNet(BasicBlock, [3, 4, 6, 3], input_channel, hidden_dim, bit_expand)


    def ResNet50(input_channel=32, hidden_dim=128, bit_expand=2):
        return ResNet(Bottleneck, [3, 4, 6, 3], input_channel, hidden_dim, bit_expand)


    def ResNet101(input_channel=32, hidden_dim=128, bit_expand=2):
        return ResNet(Bottleneck, [3, 4, 23, 3], input_channel, hidden_dim, bit_expand)


    def ResNet152(input_channel=32, hidden_dim=128, bit_expand=2):
        return ResNet(Bottleneck, [3, 8, 36, 3], input_channel, hidden_dim, bit_expand)


    class Decoder(nn.Module):
        def __init__(self, output_dim, hidden_dim):
            super(Decoder, self).__init__()
            linear_block = nn.Sequential(
                nn.Linear(hidden_dim, 2048),
                nn.BatchNorm1d(2048),
                nn.LeakyReLU(),
            )
            conv_block = nn.Sequential(
                nn.ConvTranspose2d(512, 256, 2, stride=2),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(),
                nn.ConvTranspose2d(256, 128, 2, stride=2),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(),
                nn.ConvTranspose2d(128, 64, 2, stride=2),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(),
                nn.ConvTranspose2d(64, 32, 2, stride=2),
                nn.BatchNorm2d(32),
                nn.LeakyReLU(),
                nn.ConvTranspose2d(32, 16, 2, stride=2),
                nn.BatchNorm2d(16),
                nn.LeakyReLU(),
            )

            self.linear_block = linear_block
            self.conv_block = conv_block

            self.head = nn.Conv2d(16, output_dim, kernel_size=3, stride=2, padding=1)
            self.sigmoid = nn.Sigmoid()
            self.pam = PAM_Module(in_dim=output_dim)
            self.cam = CAM_Module(in_dim=output_dim)
            self.layer_norm = nn.BatchNorm2d(output_dim)

        def forward(self, input):
            output = self.linear_block(input)
            output = output.view(-1, 512, 2, 2)
            output = self.conv_block(output) # [batch, 16, 64, 64]
            output = self.head(output)  # [batch, 3, 32, 32]
            output = output * self.sigmoid(self.pam(output) + self.cam(output))
            output = self.layer_norm(output)
            output = self.sigmoid(output)
            return output


    class ResAE(nn.Module):
        def __init__(self, latent_dim=48, channel=3, bit_expand=2):
            super(ResAE, self).__init__()
            self.encoder = ResNet34(channel, latent_dim, bit_expand)
            self.decoder = Decoder(channel, latent_dim)
            self.b_num = bit_expand

        def forward(self, x):
            # x : batch, 3, 32, 32
            code = self.encoder(x)
            recon = self.decoder(code)
            return code, recon

        def sample(self, size):
            # noise = torch.randint(2, (size, self.latent_dim)).float().to(device)
            noise = torch.randn((size, self.latent_dim)).float().to(device)
            recon = self.decoder(noise)
            return recon
        
        def bit2num(self, bit):
            b_num = self.b_num
            bit_ = bit.type(torch.float32)
            bit_ = torch.reshape(bit_, [-1, int(bit_.shape[1] / b_num), b_num])
            num = torch.zeros(bit_[:, :, 1].shape).cuda()
            for i in range(b_num):
                num = num + bit_[:, :, i] * 2 ** (b_num - 1 - i)
            num = (num + 0.5) / 2 ** b_num
            return num
            
        
        def loss(self, recon, x):
            # x : batch, 3, 32, 32
            # recon : batch, 3, 32, 32
            recon_loss = F.binary_cross_entropy(recon, x)
            return recon_loss
    return ResAE



def origin_no_bit_model():
    import math
    import torch
    import numpy as np
    import torch.nn as nn
    from torch.nn import Parameter
    import torch.nn.functional as F

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Position Attention Module
    class PAM_Module(nn.Module):
        """ Position attention module"""

        # Ref from SAGAN
        def __init__(self, in_dim):
            super(PAM_Module, self).__init__()
            self.chanel_in = in_dim
            down_scale_dim = max(in_dim // 8, 3)
            self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=down_scale_dim, kernel_size=1)
            self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=down_scale_dim, kernel_size=1)
            self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
            self.gamma = Parameter(torch.zeros(1))

            self.softmax = nn.Softmax(dim=-1)

        def forward(self, x):
            m_batchsize, C, height, width = x.size()
            proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
            proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
            energy = torch.bmm(proj_query, proj_key)
            attention = self.softmax(energy)
            proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)

            out = torch.bmm(proj_value, attention.permute(0, 2, 1))
            out = out.view(m_batchsize, C, height, width)

            out = self.gamma * out
            return out


    # Channel Attention Module
    class CAM_Module(nn.Module):
        """ Channel attention module"""

        def __init__(self, in_dim):
            super(CAM_Module, self).__init__()
            self.chanel_in = in_dim

            self.gamma = Parameter(torch.zeros(1))
            self.softmax = nn.Softmax(dim=-1)

        def forward(self, x):
            m_batchsize, C, height, width = x.size()
            proj_query = x.view(m_batchsize, C, -1)
            proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
            energy = torch.bmm(proj_query, proj_key)
            energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
            attention = self.softmax(energy_new)
            proj_value = x.view(m_batchsize, C, -1)

            out = torch.bmm(attention, proj_value)
            out = out.view(m_batchsize, C, height, width)

            out = self.gamma * out
            return out


    # ResNet Block
    class BasicBlock(nn.Module):
        expansion = 1
        def __init__(self, in_planes, planes, stride=1):
            super(BasicBlock, self).__init__()
            self.conv1 = nn.Conv2d(
                in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                                stride=1, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(planes)

            self.shortcut = nn.Sequential()
            if stride != 1 or in_planes != self.expansion*planes:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion*planes,
                            kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion*planes)
                )

        def forward(self, x):
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out += self.shortcut(x)
            out = F.relu(out)
            return out

    # ResNet BottleNeck
    class Bottleneck(nn.Module):
        expansion = 4
        def __init__(self, in_planes, planes, stride=1):
            super(Bottleneck, self).__init__()
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                                stride=stride, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(planes)
            self.conv3 = nn.Conv2d(planes, self.expansion *
                                planes, kernel_size=1, bias=False)
            self.bn3 = nn.BatchNorm2d(self.expansion*planes)

            self.shortcut = nn.Sequential()
            if stride != 1 or in_planes != self.expansion*planes:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion*planes,
                            kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion*planes)
                )

        def forward(self, x):
            out = F.relu(self.bn1(self.conv1(x)))
            out = F.relu(self.bn2(self.conv2(out)))
            out = self.bn3(self.conv3(out))
            out += self.shortcut(x)
            out = F.relu(out)
            return out

    # ResNet Backbone As Encoder
    class ResNet(nn.Module):
        def __init__(self, block, num_blocks, input_channel=1, hidden_dim=128):
            super(ResNet, self).__init__()
            self.in_planes = 64
            self.input_channel = input_channel

            self.conv1 = nn.Conv2d(input_channel, 64, kernel_size=3,
                                stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
            self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
            self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
            self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
            
            self.feat = nn.Sequential(nn.Linear(512*block.expansion, hidden_dim),
                                      nn.BatchNorm1d(hidden_dim, eps=1e-6))
            # delete batch_normal and bit layer

            self.sigmoid = nn.Sigmoid()
            

        def _make_layer(self, block, planes, num_blocks, stride):
            strides = [stride] + [1]*(num_blocks-1)
            layers = []
            for stride in strides:
                layers.append(block(self.in_planes, planes, stride))
                self.in_planes = planes * block.expansion
            return nn.Sequential(*layers)

        def forward(self, x):
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
            out = F.avg_pool2d(out, 4)
            out = out.view(out.size(0), -1)
            out = self.feat(out)
            out = self.sigmoid(out)
            
            return out


    def ResNet18(input_channel=32, hidden_dim=128):
        return ResNet(BasicBlock, [2, 2, 2, 2], input_channel, hidden_dim)


    def ResNet34(input_channel=32, hidden_dim=128):
        return ResNet(BasicBlock, [3, 4, 6, 3], input_channel, hidden_dim)


    def ResNet50(input_channel=32, hidden_dim=128):
        return ResNet(Bottleneck, [3, 4, 6, 3], input_channel, hidden_dim)


    def ResNet101(input_channel=32, hidden_dim=128):
        return ResNet(Bottleneck, [3, 4, 23, 3], input_channel, hidden_dim)


    def ResNet152(input_channel=32, hidden_dim=128):
        return ResNet(Bottleneck, [3, 8, 36, 3], input_channel, hidden_dim)


    class Decoder(nn.Module):
        def __init__(self, output_dim, hidden_dim):
            super(Decoder, self).__init__()
            linear_block = nn.Sequential(
                nn.Linear(hidden_dim, 2048),
                nn.BatchNorm1d(2048),
                nn.LeakyReLU(),
            )
            conv_block = nn.Sequential(
                nn.ConvTranspose2d(512, 256, 2, stride=2),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(),
                nn.ConvTranspose2d(256, 128, 2, stride=2),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(),
                nn.ConvTranspose2d(128, 64, 2, stride=2),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(),
                nn.ConvTranspose2d(64, 32, 2, stride=2),
                nn.BatchNorm2d(32),
                nn.LeakyReLU(),
                nn.ConvTranspose2d(32, 16, 2, stride=2),
                nn.BatchNorm2d(16),
                nn.LeakyReLU(),
            )

            self.linear_block = linear_block
            self.conv_block = conv_block

            self.head = nn.Conv2d(16, output_dim, kernel_size=3, stride=2, padding=1)
            self.sigmoid = nn.Sigmoid()
            self.pam = PAM_Module(in_dim=output_dim)
            self.cam = CAM_Module(in_dim=output_dim)
            self.layer_norm = nn.BatchNorm2d(output_dim)

        def forward(self, input):
            output = self.linear_block(input)
            output = output.view(-1, 512, 2, 2)
            output = self.conv_block(output) # [batch, 16, 64, 64]
            output = self.head(output)  # [batch, 3, 32, 32]
            output = output * self.sigmoid(self.pam(output) + self.cam(output))
            output = self.layer_norm(output)
            output = self.sigmoid(output)
            return output



    class ResAE(nn.Module):
        def __init__(self, latent_dim=48, channel=3):
            super(ResAE, self).__init__()
            self.encoder = ResNet34(channel, latent_dim)
            self.decoder = Decoder(channel, latent_dim)

        def forward(self, x):
            # x : batch, 3, 32, 32
            code = self.encoder(x)
            recon = self.decoder(code)
            return code, recon

        def sample(self, size):
            noise = torch.randint(2, (size, self.latent_dim)).float().to(device)
            recon = self.decoder(noise)
            return recon
        
        def loss(self, recon, x):
            # x : batch, 3, 32, 32
            # recon : batch, 3, 32, 32
            recon_loss = F.binary_cross_entropy(recon, x)
            return recon_loss
    return ResAE



def train(epoch, lr, batch_size, name):
    import time
    import pathlib
    import os
    cur_t = time.strftime('%Y-%m-%d', time.localtime())
    log_path = pathlib.Path(f'./{name}_{cur_t}_log')
    checkpoint_path = pathlib.Path(f'./{name}_{cur_t}_checkpoint')
    sample_path = pathlib.Path(f'./{name}_{cur_t}_samples')

    log_path.mkdir(exist_ok=True)
    checkpoint_path.mkdir(exist_ok=True)
    sample_path.mkdir(exist_ok=True)

    writer = SummaryWriter(str(log_path))
    os.system(f'cp ./*.py {str(log_path)}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = origin_bit_model()(48, 3, bit_expand=2)
    # model = origin_no_bit_model()(48, 3)
    model.cuda()

    optimizer = Adamax(model.parameters(), lr=lr)
    step_lr = StepLR(optimizer, 20, 0.5)
    train_loader, val_loader = cifar10_data_loader(batch_size=batch_size)
    batches_done = 0
    for i in range(epoch):
        # train_loss, val_loss = 0, 0
        model.train()
        for x, y in tqdm.tqdm(train_loader):
            batches_done += 1
            x = x.float().to(device)
            y = y.to(device)
            model.zero_grad()

            z, recon = model(x)
            loss = model.loss(recon, x)

            loss.backward()
            optimizer.step()

            # train_loss += loss.item() / len(train_loader)
            writer.add_scalar('loss/train', float(loss), batches_done)
    

        step_lr.step()

        with torch.no_grad():
            model.eval()
            save_flag = True
            for x, y in tqdm.tqdm(val_loader):
                x = x.float().to(device)
                y = y.to(device)
                z, recon = model(x)  # [N, 1, 28, 28]
                
                if save_flag:
                    save_image(torch.concat([x, recon], dim=0), str(sample_path / f'{i}.png'))
                    save_flag = False

                val_loss = model.loss(recon, x)
                writer.add_scalar('loss/val', float(val_loss), batches_done)
        
        # save model
        torch.save({
            'state_dict': model.state_dict(),
            'epoch': i,
            'args': arg,
            'optimizer': optimizer,
        }, str(checkpoint_path / f'model_{i}.pth.tar'))
            


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, help='the train corresponding name', default='untitled')
    parser.add_argument('--epoch', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    arg = parser.parse_args()

    train(arg.epoch, arg.lr, arg.batch_size, arg.name)
    
    
