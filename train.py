import tqdm
import torch
import torch.nn as nn
from models import origin_bit_model
from torch.optim import Adam
from data import cifar10_data_loader, tinyImageNet_data_loader, svhn_data_loader, lsun_data_loader
from losses import recon_loss, sim_loss, vae_loss
from torch.optim.lr_scheduler import StepLR
from eval import Evaluator
from ood import ReconPostProcessor
from torch.utils.tensorboard import SummaryWriter

torch.cuda.set_device(0)


writer = SummaryWriter(log_dir='./log')

def train(epochs, batch_size, lr):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, val_loader = cifar10_data_loader(batch_size=batch_size)
    # ood_loader = [tinyImageNet_data_loader(), *svhn_data_loader(), lsun_data_loader()]
    ood_loader = [lsun_data_loader()]
    # add ood loader
    

    model = origin_bit_model()(48, 3, 2) # 比特数，图像通道数，多少个bit代表一个维度
    model = model.to(device)
    # model = nn.DataParallel(model).to(device)

    # define evaluator
    evaluator = Evaluator(model)
    conf_processor = ReconPostProcessor()

    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=500, gamma=.5)

    best_loss = 10.
    for epoch in range(epochs):
        train_loss, val_loss = 0, 0
        model.train()
        for x, y in tqdm.tqdm(train_loader):
            x = x.float().to(device)
            y = y.to(device)
            model.zero_grad()

            z, recon, mu, log_var = model(x)


            ############################
            # Loss Computation
            ############################
            loss, recon_loss, kld_loss = vae_loss(x, recon, mu, log_var, 0.001)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() / len(train_loader)

        scheduler.step()
        
        # eval the ood model performance
        if epoch % 2 == 0:
            result = evaluator.eval_ood(val_loader, ood_data_loaders=ood_loader, post_processor=conf_processor, method='full')
            writer.add_scalars('train', result, epoch)

            print('Epoch : %05d | FPR@95 : %.6f | AUROC : %.6f | AUPR_IN %.6f | AUPR_OUT %.6f | Loss %.6f' \
                % (epoch, result['FPR@95'], result['AUROC'], result['AUPR_IN'], result['AUPR_OUT'], float(loss)))

        model.eval()
        with torch.no_grad():
            for x, y in tqdm.tqdm(val_loader):
                x = x.float().to(device)

                z, recon, mu, log_var = model(x)
                loss = vae_loss(x, recon, mu, log_var, 0.001)
                
                val_loss += loss.item() / len(val_loader)

            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(model.state_dict(), 'saved_models/cifar100_ae/epoch_%d_val_recon_%.6f.pth' % (epoch, val_loss))

            print('Epoch : %05d | train recon loss : %.6f | val recon loss : %.6f' % (epoch, train_loss, val_loss))


if __name__ == '__main__':
    epochs = 1000
    batch_size = 32
    lr = 1e-3
    train(epochs, batch_size, lr)
