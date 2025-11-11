import os
import torch
import torch.distributed as dist

from flow_matching_utils.flow_matching_holder import FlowMatchingRunner
from utils.util import set_seed
from config import create_config
from utils.setup_ddp import setup_ddp

config = create_config()
config.checkpoints_prefix = "ProtFlow"

config.local_rank = setup_ddp()
config.training.batch_size_per_gpu = config.training.batch_size // dist.get_world_size()
config.device = f"cuda:{dist.get_rank()}"
config.project_name = 'ProtFlow'

seed = config.seed
set_seed(seed)
if dist.get_rank() == 0:
    print(config)

flow_matching = FlowMatchingRunner(config, latent_mode=config.model.embeddings_type)

seed = config.seed + dist.get_rank()
set_seed(seed)
flow_matching.test(ckpt, test_num)