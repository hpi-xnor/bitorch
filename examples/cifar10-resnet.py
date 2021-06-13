import argparse
import logging
import sys

from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torch.nn import CrossEntropyLoss

from train import train_model

sys.path.append("../")

from bitorch.models.resnet import create_resnet


def main(args: argparse.Namespace) -> None:
    train_dataset = CIFAR10(root='./train', train=True, transform=ToTensor(), download=True)
    test_dataset = CIFAR10(root='./test', train=False, transform=ToTensor(), download=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    if args.logging is None:
        logging.basicConfig(format='%(asctime)s - %(levelname)s: %(message)s', level=logging.DEBUG, force=True)
    else:
        logging.basicConfig(filename=args.logging, format='%(asctime)s - %(levelname)s: %(message)s',
                            level=logging.DEBUG, force=True)

    logging.info("starting model training...")
    model = create_resnet(args.resnet_version, args.resnet_num_layers,
                          classes=10, initial_layers="mnist", image_channels=3)
    train_model(model, train_loader, test_loader, CrossEntropyLoss(), epochs=args.epochs,
                lr=args.lr, log_interval=args.log_interval, gpu=args.cuda)
    logging.info("model training finished!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Bitorch RenNet Imagenet Example')
    parser.add_argument('--batch-size', type=int, default=100,
                        help='batch size for training and testing (default: 100)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='Train on GPU with CUDA')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument("-l", "--logging", type=str, required=False, default=None,
                        help="output file path for logging. default to stdout")
    parser.add_argument("--resnet-version", type=int, choices=[1, 2], required=True,
                        help="version of resnet to be used")
    parser.add_argument("--resnet-num-layers", type=int, choices=[18, 34, 50, 152], required=True,
                        help="number of layers to be used inside resnet")

    args = parser.parse_args()

    main(args)
