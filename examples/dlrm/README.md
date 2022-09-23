# Pytorch Lightning Example Script

To give an example on how to use bitorch for your own recommendation projects `train_dlrm.py` trains a quantized version of Facebooks [DLRM](https://github.com/facebookresearch/dlrm) implemented in `bitorch` on an ad recommendation dataset.
Right now only the [Criteo Ad Challenge](https://labs.criteo.com/2014/02/kaggle-display-advertising-challenge-dataset/) dataset is supported.

First the requirements for this example need to be installed
(unless the optional dependencies of BITorch were already installed):
```bash
pip install -r requirements.txt
```

Below you can find an example call of the script:
```bash
python examples/dlrm/train_dlrm.py  --dataset criteo --input-quantization sign --weight-quantization approxsign --download --ignore-dataset-size 0.0 --batch-size 8192 --lr-scheduler cosine --optimizer adam --wandb --batch-size-test 10000 --num-workers 0 --dataset-dir /datasets --gpus 1 --max_epochs 10
```

If the dataset is not present in the given directory, it will be downloaded to the specified directory and preprocessed. Preprocessing usually takes about 30 min, depending on your hardware setup.

## Arguments

To find an exhaustive overview over the parameters to configure the `train_dlrm.py` script, call `python train_dlrm.py --help`.
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

### dataset args

- `--datset` name of dataset to train on. Chose from `criteo`
- `--download` toggles if dataset if not present at `--dataset-dir` should be downloaded.
- `--dataset-dir` path to dataset.
- `--num-worker` sets number of workers for dataloading

### quantization args

- `--input-quantization` chooses the default input quantization method.
- `--weight-quantization` chooses the default weight quantization method.
- `--gradient-cancellation-threshold` sets the default gradient cancellation threshold
