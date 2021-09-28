import argparse
import logging
from typing import Dict

from bitorch.datasets import dataset_from_name
from bitorch.datasets.imagenet import ImageNet
from bitorch.models import model_from_name
from examples.image_classification.utils.arg_parser import create_argparser


import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torch_xla
import torch_xla.debug.metrics as met
import torch_xla.distributed.parallel_loader as pl
import torch_xla.utils.utils as xu
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.test.test_utils as test_utils

from examples.image_classification.utils.utils import create_optimizer, create_scheduler


def _train_update(device, step, loss, tracker, epoch, writer):
    test_utils.print_training_update(
        device,
        step,
        loss.item(),
        tracker.rate(),
        tracker.global_rate(),
        epoch,
        summary_writer=writer)


def main(args: argparse.Namespace, model_kwargs: Dict) -> None:
    """trains a model on the configured image dataset with tpu.

    Args:
        args (argparse.Namespace): cli arguments
        model_kwargs (dict): model specific cli arguments as a dictionary
    """
    print('==> Preparing data..')
    dataset = dataset_from_name(args.dataset)
    assert dataset == ImageNet, "TPU training should only be used to train imagenet."

    img_dim = dataset.shape[-1]
    if args.fake_data:
        train_loader = xu.SampleGenerator(
            data=(torch.zeros(args.batch_size, 3, img_dim, img_dim),
                  torch.zeros(args.batch_size, dtype=torch.int64)),
            sample_count=ImageNet.num_train_samples // args.batch_size // xm.xrt_world_size())
        test_loader = xu.SampleGenerator(
            data=(torch.zeros(args.batch_size, 3, img_dim, img_dim),
                  torch.zeros(args.batch_size, dtype=torch.int64)),
            sample_count=ImageNet.num_val_samples // args.batch_size // xm.xrt_world_size())
    else:
        train_dataset = torchvision.datasets.ImageFolder(
            os.path.join(args.dataset_dir, 'train'),
            ImageNet.train_transform())
        assert ImageNet.num_train_samples == len(train_dataset.imgs), "not all imagenet images are present"
        test_dataset = torchvision.datasets.ImageFolder(
            os.path.join(args.dataset_dir, 'val'),
            ImageNet.test_transform())

        train_sampler, test_sampler = None, None
        if xm.xrt_world_size() > 1:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_dataset,
                num_replicas=xm.xrt_world_size(),
                rank=xm.get_ordinal(),
                shuffle=True)
            test_sampler = torch.utils.data.distributed.DistributedSampler(
                test_dataset,
                num_replicas=xm.xrt_world_size(),
                rank=xm.get_ordinal(),
                shuffle=False)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=train_sampler,
            shuffle=False if train_sampler else True,
            num_workers=args.num_workers)
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            sampler=test_sampler,
            shuffle=False,
            num_workers=args.num_workers)

    torch.manual_seed(42)

    device = xm.xla_device()
    model = model_from_name(args.model)(**model_kwargs, dataset=dataset).to(device)  # type: ignore
    writer = None
    if xm.is_master_ordinal():
        writer = test_utils.get_summary_writer("./tblogs")
    optimizer = create_optimizer(args.optimizer, model, args.lr, args.momentum)
    lr_scheduler = create_scheduler(args.lr_scheduler, optimizer, args.lr_factor, args.lr_steps, args.epochs)

    loss_fn = nn.CrossEntropyLoss()

    def train_loop_fn(loader, epoch):
        tracker = xm.RateTracker()
        model.train()
        for step, (data, target) in enumerate(loader):
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            xm.optimizer_step(optimizer)

            tracker.add(args.batch_size)
            if lr_scheduler:
                lr_scheduler.step()
            if step % args.log_interval == 0:
                xm.add_step_closure(
                    _train_update, args=(device, step, loss, tracker, epoch, writer))

    def test_loop_fn(loader, current_epoch):
        total_samples, correct = 0, 0
        model.eval()
        for step, (data, target) in enumerate(loader):
            output = model(data)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum()
            total_samples += data.size()[0]
            if step % args.log_interval == 0:
                xm.add_step_closure(
                    test_utils.print_test_update, args=(device, None, current_epoch, step))
        acc = 100.0 * correct.item() / total_samples
        acc = xm.mesh_reduce('test_accuracy', acc, np.mean)
        return acc

    train_device_loader = pl.MpDeviceLoader(train_loader, device)
    test_device_loader = pl.MpDeviceLoader(test_loader, device)
    accuracy, max_accuracy = 0.0, 0.0
    for epoch in range(1, args.epochs + 1):
        xm.master_print('Epoch {} train begin {}'.format(epoch, test_utils.now()))
        train_loop_fn(train_device_loader, epoch)
        xm.master_print('Epoch {} train end {}'.format(epoch, test_utils.now()))
        accuracy = test_loop_fn(test_device_loader, epoch)
        xm.master_print('Epoch {} test end {}, Accuracy={:.2f}'.format(
            epoch, test_utils.now(), accuracy))
        max_accuracy = max(accuracy, max_accuracy)
        test_utils.write_to_summary(
            writer,
            epoch,
            dict_to_write={'Accuracy/test': accuracy},
            write_xla_metrics=True)

    test_utils.close_summary_writer(writer)
    xm.master_print('Max Accuracy: {:.2f}%'.format(max_accuracy))
    return max_accuracy


def _mp_fn(index, args, model_kwargs):
    torch.set_default_tensor_type('torch.FloatTensor')
    main(args, model_kwargs)


if __name__ == '__main__':
    parser, model_parser = create_argparser()
    parser.add_argument("--fake-data", action="store_true",
                        help="train with fake data")
    parser.add_argument("--num-procs", default=None, type=int,
                        help="set a number of tpu processors")
    args, unparsed_model_args = parser.parse_known_args()
    model_args = model_parser.parse_args(unparsed_model_args)

    model_kwargs = vars(model_args)
    print(f"got model args as dict: {model_kwargs}")
    logging.disable()

    xmp.spawn(_mp_fn, args=(args, model_kwargs), nprocs=args.num_procs)
