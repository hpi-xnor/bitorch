"""Args from PyTorch Lightning's Trainer that are currently unused and a function to deal with them."""
from argparse import Namespace
from typing import List


def clean_hyperparameters(args: Namespace) -> Namespace:
    """Remove args which are not passed to the constructor in our training script."""
    clean_args = Namespace()
    for key in args.__dict__.keys():
        if key in UNUSED_PL_ARGS:
            continue
        setattr(clean_args, key, getattr(args, key))
    return clean_args


# this list is copied from the constructor of PyTorch's Trainer, but all arguments used in our script were removed
UNUSED_PL_ARGS: List[str] = [
    "logger",
    "checkpoint_callback",
    "enable_checkpointing",
    "callbacks",
    "default_root_dir",
    "gradient_clip_val",
    "gradient_clip_algorithm",
    "process_position",
    "num_nodes",
    "num_processes",
    "devices",
    "auto_select_gpus",
    "tpu_cores",
    "ipus",
    "log_gpu_memory",
    "progress_bar_refresh_rate",
    "enable_progress_bar",
    "overfit_batches",
    "track_grad_norm",
    "check_val_every_n_epoch",
    "fast_dev_run",
    "accumulate_grad_batches",
    "min_epochs",
    "min_steps",
    "max_time",
    "limit_train_batches",
    "limit_val_batches",
    "limit_test_batches",
    "limit_predict_batches",
    "val_check_interval",
    "flush_logs_every_n_steps",
    "log_every_n_steps",
    "sync_batchnorm",
    "precision",
    "enable_model_summary",
    "weights_summary",
    "weights_save_path",
    "num_sanity_val_steps",
    "resume_from_checkpoint",
    "profiler",
    "benchmark",
    "deterministic",
    "reload_dataloaders_every_n_epochs",
    "auto_lr_find",
    "replace_sampler_ddp",
    "detect_anomaly",
    "auto_scale_batch_size",
    "prepare_data_per_node",
    "plugins",
    "amp_backend",
    "amp_level",
    "move_metrics_to_cpu",
    "multiple_trainloader_mode",
    "stochastic_weight_avg",
    "terminate_on_nan",
]
