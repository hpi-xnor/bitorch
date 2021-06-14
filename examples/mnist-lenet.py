import argparse
import logging
import sys

from torchvision.datasets import mnist
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torch.nn import CrossEntropyLoss

from train import train_model

sys.path.append("../")

from bitorch.models.lenet import LeNet


def main(args: argparse.Namespace) -> None:

    train_dataset = mnist.MNIST(root='./train', train=True, transform=ToTensor(), download=True)
    test_dataset = mnist.MNIST(root='./test', train=False, transform=ToTensor(), download=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    if args.logging is None:
        logging.basicConfig(format='%(asctime)s - %(levelname)s: %(message)s', level=logging.DEBUG, force=True)
    else:
        logging.basicConfig(filename=args.logging, format='%(asctime)s - %(levelname)s: %(message)s',
                            level=logging.DEBUG, force=True)

    logging.info("starting model training...")
    model_type = 'full_precision' if args.bits == 32 else 'quantized'
    train_model(LeNet(model_type), train_loader, test_loader, CrossEntropyLoss(), epochs=args.epochs,
                lr=args.lr, log_interval=args.log_interval, gpu=args.cuda)
    logging.info("model training finished!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Bitorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=100,
                        help='batch size for training and testing (default: 100)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='Train on GPU with CUDA')
    parser.add_argument('--bits', type=int, default=32, required=False,
                        help='Number of bits for binarization/quantization')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument("-l", "--logging", type=str, required=False, default=None,
                        help="output file path for logging. default to stdout")

    args = parser.parse_args()

    main(args)
