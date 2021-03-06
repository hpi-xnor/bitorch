# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).

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
