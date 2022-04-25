# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).

## Unreleased

### Added

- quantization bit width information to quantization functions

### Fixed

- a bug where layer input and weight quantization functions could not be set using command line arguments

## [0.1.1] - 2022/01/21

### Changed

- make package compatible with python 3.7
- added quantization bit width information to quantization functions

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
