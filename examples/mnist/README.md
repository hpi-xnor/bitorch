# Example for MNIST

In this example script we train a simple model for the MNIST dataset and also use the [bitorch inference engine](https://github.com/hpi-xnor/bitorch-inference-engine) for speed up.

First the requirements for this example need to be installed
(unless the optional dependencies of BITorch were already installed):
```bash
pip install -r requirements.txt
```

Then you can run the following to train an MLP with 3 layers (one of which is a binary layer),
or add `--help` for more arguments:
```bash
python train_mnist.py --epochs 10 --model mlp --log-interval 100
```
