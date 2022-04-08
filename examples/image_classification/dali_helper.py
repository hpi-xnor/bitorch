from typing import Any, List, Tuple
from pathlib import Path
import logging

try:
    from nvidia.dali.plugin.pytorch import DALIClassificationIterator
    from nvidia.dali.pipeline import Pipeline
    import nvidia.dali.ops as ops
    import nvidia.dali.types as types
except ImportError:
    raise ImportError("NV-DALI has not been installed yet, please install DALI first "
                      "(https://www.github.com/NVIDIA/DALI).")


class HybridTrainPipe(Pipeline):
    """TODO: there are some magic numbers in this class, we should provide more comments for them,
    or even find a better solution?
    """

    def __init__(
            self,
            batch_size: int,
            num_threads: int,
            rank: int,
            world_size: int,
            data_dir: str,
            crop: int,
            dali_cpu: bool = True) -> None:
        super(HybridTrainPipe, self).__init__(batch_size, num_threads, rank, seed=12 + rank,
                                              set_affinity=True)
        self.input = ops.FileReader(file_root=data_dir, num_shards=world_size, shard_id=rank,
                                    random_shuffle=True)

        dali_device = 'cpu' if dali_cpu else 'gpu'
        decoder_device = 'cpu' if dali_cpu else 'mixed'
        device_memory_padding = 211025920 if decoder_device == 'mixed' else 0
        host_memory_padding = 140544512 if decoder_device == 'mixed' else 0
        self.decode = ops.ImageDecoderRandomCrop(device=decoder_device, output_type=types.RGB,
                                                 device_memory_padding=device_memory_padding,
                                                 host_memory_padding=host_memory_padding,
                                                 random_aspect_ratio=[0.8, 1.25],
                                                 random_area=[0.08, 1.0],
                                                 num_attempts=100)

        self.res = ops.Resize(device=dali_device, resize_x=crop, resize_y=crop, interp_type=types.INTERP_TRIANGULAR)

        self.cmnp = ops.CropMirrorNormalize(device=dali_device,
                                            output_dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            crop=(crop, crop),
                                            image_type=types.RGB,
                                            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                                            std=[0.229 * 255, 0.224 * 255, 0.225 * 255])
        self.coin = ops.CoinFlip(probability=0.5)
        print('DALI "{0}" variant'.format(dali_device))

    def define_graph(self) -> List[Any]:
        rng = self.coin()
        self.jpegs, self.labels = self.input(name="Reader")
        images = self.decode(self.jpegs)
        images = self.res(images)
        output = self.cmnp(images.gpu(), mirror=rng)

        return [output, self.labels]


class HybridValPipe(Pipeline):
    """
    TODO: add description here!
    """
    def __init__(
            self,
            batch_size: int,
            num_threads: int,
            rank: int,
            world_size: int,
            data_dir: str,
            crop: int,
            size: int,
            dali_cpu: bool = True) -> None:
        super(HybridValPipe, self).__init__(batch_size, num_threads, rank, seed=12 + rank, set_affinity=True)

        dali_device = 'cpu' if dali_cpu else 'gpu'
        decoder_device = 'cpu' if dali_cpu else 'mixed'
        self.input = ops.FileReader(file_root=data_dir, num_shards=world_size, shard_id=rank,
                                    random_shuffle=False)
        self.decode = ops.ImageDecoder(device=decoder_device, output_type=types.RGB)
        # self.decode = ops.nvJPEGDecoder(device="mixed", output_type=types.RGB)
        self.res = ops.Resize(device=dali_device, resize_shorter=size, interp_type=types.INTERP_TRIANGULAR)
        self.cmnp = ops.CropMirrorNormalize(device=dali_device,
                                            output_dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            crop=(crop, crop),
                                            image_type=types.RGB,
                                            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                                            std=[0.229 * 255, 0.224 * 255, 0.225 * 255])

    def define_graph(self) -> List[Any]:
        self.jpegs, self.labels = self.input(name="Reader")
        images = self.decode(self.jpegs)
        images = self.res(images)
        output = self.cmnp(images)
        # self.iteration += 1
        # if self.iteration % 200 == 0:
        #  del images, self.jpegs
        return [output, self.labels]


def create_dali_data_loader(
        train_dir: Path,
        test_dir: Path,
        dali_gpu_id: int,
        world_size: int,
        dali_cpu: int,
        batch_size: int,
        num_workers: int,
        img_crop_size: int = 224,
        img_size_val: int = 256) -> Tuple[DALIClassificationIterator, DALIClassificationIterator]:
    """creates nv-dali dataloader for image classification.
    :param  input args
    :param img_crop_size: image crop size
    :param img_size_val: image size for validation
    :return: data loader for training and validation
    """
    logging.debug("creating dali pipelines....")
    pipe = HybridTrainPipe(batch_size=batch_size, num_threads=num_workers, rank=dali_gpu_id, world_size=world_size,
                           data_dir=train_dir, crop=img_crop_size, dali_cpu=dali_cpu)
    pipe.build()
    train_loader = DALIClassificationIterator(pipe, size=int(pipe.epoch_size("Reader")), auto_reset=True)

    pipe = HybridValPipe(batch_size=batch_size, num_threads=num_workers, rank=dali_gpu_id, world_size=world_size,
                         data_dir=test_dir, crop=img_crop_size, size=img_size_val, dali_cpu=dali_cpu)
    pipe.build()
    test_loader = DALIClassificationIterator(pipe, size=int(pipe.epoch_size("Reader")), auto_reset=True)

    return train_loader, test_loader
