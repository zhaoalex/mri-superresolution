from __future__ import print_function

from math import log10

import torch
import torch.backends.cudnn as cudnn

from SRCNN.model import Net
from progress_bar import progress_bar
from numpy import argmax
from shutil import copyfile

class SRCNNTrainer(object):
    def __init__(self, config, training_loader, testing_loader):
        super(SRCNNTrainer, self).__init__()
        self.CUDA = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.CUDA else 'cpu')
        self.model = None
        self.lr = config.lr
        self.nEpochs = config.nEpochs
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        self.seed = config.seed
        self.upscale_factor = config.upscale_factor
        self.training_loader = training_loader
        self.testing_loader = testing_loader

        self.load = config.load
        self.model_path = 'models/SRCNN/' + str(self.upscale_factor)

    def build_model(self):
        if self.load:
            self.model = torch.load(self.model_path + '/' + self.load)
            print('===> Model loaded')
        else:
            self.model = Net(num_channels=1, base_filter=64, upscale_factor=self.upscale_factor).to(self.device)
            self.model.weight_init(mean=0.0, std=0.01)
        
        self.criterion = torch.nn.MSELoss()
        torch.manual_seed(self.seed)

        if self.CUDA:
            torch.cuda.manual_seed(self.seed)
            cudnn.benchmark = True
            self.criterion.cuda()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[50, 75, 100], gamma=0.5)

    def save_model(self, epoch):
        model_out_path = self.model_path + "/model_{}.pth".format(epoch)
        torch.save(self.model, model_out_path)
        print("Checkpoint saved to {}".format(model_out_path))

    def train(self):
        self.model.train()
        train_loss = 0
        for batch_num, (data, target) in enumerate(self.training_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            loss = self.criterion(self.model(data), target)
            train_loss += loss.item()
            loss.backward()
            self.optimizer.step()
            progress_bar(batch_num, len(self.training_loader), 'Loss: %.4f' % (train_loss / (batch_num + 1)))

        print("    Average Loss: {:.4f}".format(train_loss / len(self.training_loader)))

    def test(self):
        self.model.eval()
        avg_psnr = 0

        with torch.no_grad():
            for batch_num, (data, target) in enumerate(self.testing_loader):
                data, target = data.to(self.device), target.to(self.device)
                prediction = self.model(data)
                mse = self.criterion(prediction, target)
                psnr = 10 * log10(1 / mse.item())
                avg_psnr += psnr
                progress_bar(batch_num, len(self.testing_loader), 'PSNR: %.4f' % (avg_psnr / (batch_num + 1)))

        print("    Average PSNR: {:.4f} dB".format(avg_psnr / len(self.testing_loader)))
        return avg_psnr / len(self.testing_loader)

    def run(self):
        self.build_model()
        all_epoch_psnrs = []
        for epoch in range(1, self.nEpochs + 1):
            print("\n===> Epoch {} starts:".format(epoch))
            self.train()
            epoch_psnr = self.test()
            all_epoch_psnrs.append(epoch_psnr)
            self.scheduler.step()
            # if epoch == self.nEpochs:
            self.save_model(epoch) # save model every 5 epochs
        
        best_epoch = argmax(all_epoch_psnrs) + 1
        print("Best epoch: model_{} with PSNR {}".format(best_epoch, all_epoch_psnrs[best_epoch - 1]))
        copyfile(self.model_path + "/model_{}.pth".format(best_epoch), self.model_path + "/best_model.pth")
