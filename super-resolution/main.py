from __future__ import print_function

import argparse

from torch.utils.data import DataLoader

from FSRCNN.solver import FSRCNNTrainer
from SRCNN.solver import SRCNNTrainer
from SRGAN.solver import SRGANTrainer
from dataset.data import get_training_set, get_test_set

# ===========================================================
# Training settings
# ===========================================================
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
# hyper-parameters
parser.add_argument('--batchSize', type=int, default=1, help='training batch size')
parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')
parser.add_argument('--nEpochs', type=int, default=20, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='Learning Rate. Default=0.001')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')


# model configuration
parser.add_argument('--upscale_factor', '-uf',  type=int, default=2, help="super resolution upscale factor")
parser.add_argument('--model', '-m', type=str, default='fsrcnn', help='choose which model is going to use')
parser.add_argument('--load', type=str, help='loaded trained model or train from scratch')

args = parser.parse_args()


def main():
    # ===========================================================
    # Set train dataset & test dataset
    # ===========================================================
    print('===> Loading datasets')
    train_set = get_training_set(args.upscale_factor)
    test_set = get_test_set(args.upscale_factor)
    training_data_loader = DataLoader(dataset=train_set, batch_size=args.batchSize, shuffle=True)
    testing_data_loader = DataLoader(dataset=test_set, batch_size=args.testBatchSize, shuffle=False)

    if args.model == 'srcnn':
        model = SRCNNTrainer(args, training_data_loader, testing_data_loader)
    elif args.model == 'fsrcnn':
        model = FSRCNNTrainer(args, training_data_loader, testing_data_loader)
    elif args.model == 'srgan':
        model = SRGANTrainer(args, training_data_loader, testing_data_loader)
    else:
        raise Exception("the model does not exist")

    model.run()


if __name__ == '__main__':
    main()
