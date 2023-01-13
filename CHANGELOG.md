# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).

## [0.3.0] - 2023/01/13

### Added

- new models:
  - [MeliusNet](bitorch/models/meliusnet.py)
  - [BinaryDenseNet](bitorch/models/densenet.py)
  - [QuickNet](bitorch/models/quicknet.py)
- simple example script for MNIST
- support for integration of bitorch's inference engine for the following layers
  - QLinear
  - QConv
- a quantized DLRM version, derived from [this](https://github.com/facebookresearch/dlrm) implementation
- example code for training the quantized DLRM model
- new quantization function: [Progressive Sign](bitorch/quantizations/progressive_sign.py)
- new features in PyTorch Lightning example:
  - training with Knowledge Distillation
  - improved logging
  - callback to update Progressive Sign module
- option to integrate custom models, datasets, quantization functions
- a quantization scheduler which lets you change quantization methods during training
- a padding layer

### Changed

- requirements changed:
  - code now depends on torch 1.12.x and torchvision 0.13.x
  - requirements for examples are now stored at their respective folders
  - optional requirements now install everything needed to run all examples
- code is now formatted with the black code formatter
- using PyTorch's implementation of RAdam
- renamed the `bitwidth` attribute of quantization functions to `bit_width`
- moved the image datasets out of the bitorch core package into the image classification example

### Fixed

- fix error from updated protobuf package

## [0.2.0] - 2022/05/19

### Added

- automatic documentation generation using sphinx
- more documentation of layers and modules
- bit-width of quantization functions is now stored
- new layers:
  - [Pact](https://arxiv.org/abs/1805.06085) activation function
  - QEmbedding
  - QEmbeddingBag
- fvbitcore support in the example scripts for flop and model size estimation on operation level

### Changed

- image classification example:
  - script now uses [pytorch lightning](https://www.pytorchlightning.ai/)
  - it includes distributed training capability
  - added wandb metric logging
- QConv layers can now be pickled
- Different quantized versions of LeNet available

### Fixed

- a bug where layer input and weight quantization functions could not be set using command line arguments
- a bug where modules could not be imported in OS that use different path separators than '/'

## [0.1.1] - 2022/01/21

### Changed

- make package compatible with python 3.7

## [0.1.0] - 2022/01/06

### Added

- basic quantized layers
  - QActivation
  - QConv
  - QLinear
- several debug layers
- resnet, lenet
- various quantization functions
  - approxsign
  - dorefa
  - sign
  - steheaviside
  - swishsign
- support for cifar10 and mnist
- general training script for image classification
- result logger for csv and tensorboard
- checkpoint manager
- eta estimator
- experiment creator
- model visualization in logs and tensorboard
- config classes for bitorch layers / quantizations with automated argparse argument creation
