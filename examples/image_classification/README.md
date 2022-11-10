# Pytorch Lightning Example Script

To give an example on how to use bitorch for your own projects `image_classification.py` trains one of the
models implemented in `bitorch` on an image classification dataset.

First the requirements for this example need to be installed
(unless the optional dependencies of BITorch were already installed):
```bash
pip install -r requirements.txt
```

Below you can find an example call of the script:
```bash
python3 image_classification.py --optimizer adam --lr 0.001 --lr-scheduler cosine --max_epochs 2 --dataset imagenet --model resnet18v1 --batch-size 128 --accelerator gpu --num-workers 16 --gpus 3
```

## Arguments

To find an exhaustive overview over the parameters to configure the `image_classification.py` script, call `python image_classification.py --help`.
The list below gives a brief overview over some selected arguments.

### general training args

- `--optimizer` sets the optimizer. Choose from `adam, sgd` and `radam`.
- `--lr-scheduler` sets the learning rate scheduler. Choose from `cosine, step` and `exponential`
- `--lr` sets the used learning rate.
- `--max-epochs` sets the number of epochs to train.
- `--max-steps` sets the number of training steps to perform.
- `--batch-size` sets batchsize to use
- `--gpus n` specify number of gpus to use. if `n` not specified, all available gpus will be used.
- `--cpu` force training on cpu.

### logging args

- `--log-file` specifies the file to log into
- `--log-stdout` toggles if the log output should also go to stdout
- `--tensorboar` toggles logging to tensorboard
- `--wandb` toggles logging to wandb. You need to specify a WANDB_API_TOKEN variable in your environment to use this. [details](https://docs.wandb.ai/guides/track/public-api-guide#authentication)
- `--result-file` specifies path to a result file which will contain the evaluation metrics in csv format.
- `--checkpoint-dir` path to where checkpoints shall be stored
- `--checkpoint-load` path to checkpoint to load from

### model args

- `--model` specify name of model you want to train. Choose from `Lenet,Resnet,Resnet152V1,Resnet152V2,Resnet18V1,Resnet18V2,Resnet34V1,Resnet34V2,Resnet50V1,Resnet50V2,ResnetE,ResnetE18,ResnetE34,Quicknet,QuicknetSmall` or `QuickNetLarge`

Each model can have specific arguments. Check them by calling `python image_classification.py --help`.

### dataset args

- `--datset` name of dataset to train on. Chose from `mnist, cifar10, cifar100` and `imagenet`
- `--download` toggles if dataset is not present at `--dataset-dir` should be downloaded. Only available for `mnist` and `cifar10`.
- `--dataset-dir` path to dataset.
- `--num-worker` sets number of workers for dataloading

### quantization args

- `--input-quantization` chooses the default input quantization method.
- `--weight-quantization` chooses the default weight quantization method.
- `--gradient-cancellation-threshold` sets the default gradient cancellation threshold
