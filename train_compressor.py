import os
import torch.distributed as dist

from compressors.hourglass_train import HourglassProteinCompressionTransformer

from utils.util import set_seed

if __name__ == '__main__':

    set_seed(2025)
    model = HourglassProteinCompressionTransformer(
        dim=480,
        checkpoints_folder="", #TBD
        use_wandb_log=False,
        max_sequence_len = 50,
        train_batch_size = 16,
        valid_batch_size = 16,
        train_epoch = 10000,
        eval_freq = 500, #5000, #in step
        logger_name = "tanh",
        depth=4,
        downproj_factor=16,
        shorten_factor=1,
        attn_resampling=True,
        updown_sample_type="naive",
        heads=8,
        dim_head=64,
        causal=False,
        norm_out=False,
        use_quantizer="tanh",
        n_e=128,
        e_dim=64,
        fsq_levels = None,
        lr=8e-5,
        lr_adam_betas=(0.9, 0.999),
        lr_sched_type = "cosine_with_restarts",
        lr_num_warmup_steps = 10000,
        lr_num_training_steps = 10_000_000,
        lr_num_cycles = 2
    )
    
    model.train()
