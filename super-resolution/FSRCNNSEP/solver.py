from __future__ import print_function
from math import log10
from os.path import exists

import torch
import torch.backends.cudnn as cudnn
from FSRCNN.model import Net
from progress_bar import progress_bar
from numpy import argmax
from shutil import copyfile
from os import makedirs
import torch.nn as nn


class FSRCNNSepTrainer(object):
    def __init__(self, config, training_loader, testing_loader):
        super(FSRCNNSepTrainer, self).__init__()
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
        self.model_path = 'models/FSRCNNSEP/' + str(self.upscale_factor)

        makedirs(self.model_path, exist_ok=True)

    def build_model(self):
        if self.load:
            self.model = torch.load('models/FSRCNNSEP/' + self.load)
            print('===> Model loaded')
        else:
            return
        #     print('===> Init new network')
        #     self.model = Net(num_channels=1, upscale_factor=self.upscale_factor).to(self.device) # only train on Y channel
        #     self.model.weight_init(mean=0.0, std=0.2)

        for child in self.model.children():
            for param in child.parameters():
                param.requires_grad = False
        
        self.model.last_part = nn.ConvTranspose2d(in_channels=self.model.first_part[0].out_channels, out_channels=self.model.first_part[0].in_channels, kernel_size=9, stride=self.upscale_factor, padding=4, output_padding=self.upscale_factor-1)
        for param in self.model.last_part.parameters():
            param.requires_grad = True

        self.criterion = torch.nn.MSELoss()
        torch.manual_seed(self.seed)

        if self.CUDA:
            torch.cuda.manual_seed(self.seed)
            cudnn.benchmark = True
            self.criterion.cuda()

        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[50, 75, 100], gamma=0.5)  # lr decay

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
            loss.backward() #compute gradient
            self.optimizer.step() #update weight
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

        with open(self.model_path + '/metrics.txt', 'w+') as metricsfile:
            print("Saving metrics")
            for i, psnr in enumerate(all_epoch_psnrs):
                metricsfile.write("{},{}\n".format(i+1, psnr))
            metricsfile.write("Best epoch: model_{} with PSNR {}\n".format(best_epoch, all_epoch_psnrs[best_epoch - 1]))
