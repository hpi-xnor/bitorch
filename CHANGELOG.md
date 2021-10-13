# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](http://keepachangelog.com/) 
and this project adheres to [Semantic Versioning](http://semver.org/).

## [0.1.0] - Unreleased

- added basic quanitzed layers
    - QActivation
    - QConv
    - QLinear
- added several debug layers
- added resnet, lenet
- added various quantization functions
    - approxsign
    - dorefa
    - sign
    - steheaviside
    - swishsign
- added support for cifar10 and mnist
- adds general training script for image classification
- result logger for csv and tensorboard
- checkpoint manager
- eta estimator
- experiment creator
- model visualization in logs and tensorboard
- config classes for bitorch layers / quantizations with automated argparse argument creation
