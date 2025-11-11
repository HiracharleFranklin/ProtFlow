import os
import json
import math
import torch
import wandb
import numpy as np
import torch.distributed as dist
from torch.distributions import LogisticNormal
from copy import deepcopy
from ml_collections import ConfigDict
from random import random
from typing import Optional, Union, Dict
from tqdm import tqdm
from tqdm.auto import trange
from torch.utils.data import DataLoader
from torch.nn.functional import cross_entropy
from torch.cuda.amp import GradScaler
from timm.scheduler.cosine_lr import CosineLRScheduler
from typing import List, Dict, Union, Tuple

from encoders import EncNormalizer, ESM2EncoderModel
from compressers import HourglassProteinCompressionTransformer, trim_or_pad_batch_first
from model.fm_estimator import FlowEstimatorEMB, FlowEstimatorEMBwithVI
from model.ema_model import ExponentialMovingAverage
from flow_matching_utils.length_sampler import LengthSampler
from flow_matching_utils.reflow_dataset import ReflowDataset
from utils import load_fasta_file, set_seed, gather_texts, dict_to_cuda, reduce_tensor, make_mask_wo_SEP_CLS, masked_mean, masked_std
from evaluation import calculate_fid_for_files

import ot as pot
import torchdyn
from torchdyn.core import NeuralODE

from torchcfm.conditional_flow_matching import *
from torchcfm.models.models import *
from torchcfm.utils import *
from torchcfm.optimal_transport import *

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('') # TBD

class FlowMatchingRunner:
    def __init__(
            self,
            config: ConfigDict,
            eval: bool = False,
            latent_mode: str = "embeddings"
    ):
        # Basic info
        self.config = config
        self.latent_mode = latent_mode
        self.eval = eval
        self.class_type = config.class_type

        self.checkpoints_folder = config.training.checkpoints_folder
    
        self.enc_normalizer = EncNormalizer(
            enc_mean_path=self.config.data.enc_mean,
            enc_std_path=self.config.data.enc_std,
            enc_max_path=self.config.data.enc_max,
            enc_min_path=self.config.data.enc_min,
        ).cuda()
        self.encoder_decoder = ESM2EncoderModel(
            self.config.model.hg_name,
            device=self.config.device,
            enc_normalizer=self.enc_normalizer,
            decoder_path=self.config.decoder_path,
            max_seq_len=self.config.data.max_sequence_len,
        )
        self.compresser = HourglassProteinCompressionTransformer(
            dim=self.config.model.hidden_size, 
            depth=self.config.compress.depth, 
            downproj_factor=self.config.compress.downproj_factor, 
            shorten_factor=self.config.compress.shorten_factor, 
            attn_resampling=self.config.compress.attn_resampling, 
            updown_sample_type=self.config.compress.updown_sample_type, 
            heads=self.config.compress.heads, 
            dim_head=self.config.compress.dim_head, 
            causal=self.config.compress.causal, 
            norm_out=self.config.compress.norm_out,
            use_quantizer=self.config.compress.use_quantizer, 
            n_e=self.config.compress.n_e, 
            e_dim=self.config.compress.e_dim, 
            vq_beta=self.config.compress.vq_beta, 
            enforce_single_codebook_per_position=self.config.compress.enforce_single_codebook_per_position, 
            fsq_levels=self.config.compress.fsq_levels
        ).cuda()
        self.compresser.from_pretrained(self.config.compress.checkpoint)

        self.optimizer = None
        self.scheduler = None
        self.sampler = None
        self.flow_matcher = None
        self.step = 0

        if self.config.use_compress:
            self.model = FlowEstimatorEMBwithVI(
                input_size=self.config.model.compressed_hidden_size,
                config=config.bert_config
            ).cuda().train()
        else:
            self.model = FlowEstimatorEMB(
                input_size=self.config.model.hidden_size,
                config=config.bert_config
            ).cuda().train()

        self.ddp_model = self.model
        if self.config.ddp:
            self.ddp_model = torch.nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[config.local_rank],
                broadcast_buffers=False,
            )
        self.total_number_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.config.model.total_number_params = self.total_number_params
        self.device = next(self.model.parameters()).device

        if eval:
            self.ema = ExponentialMovingAverage(self.model.parameters(), config.model.ema_rate)
            self.restore_parameters(self.device)
            self.switch_to_ema()
            self.model.eval()
        
        self.train_dataset = None
        self.valid_dataset = None
        self.length_sampler = LengthSampler(path=self.config.data.train_dataset_path, max_len=self.config.data.max_sequence_len - 2)
        
        
        if self.config.ddp and dist.get_rank() == 0 and self.config.wandb and not eval:
            wandb.init(
                project=self.config.project_name,
                name=self.config.checkpoints_prefix,
                config=dict(self.config),
                mode="online"
            )

        self.logistic_normal_dist = LogisticNormal(loc=torch.tensor(self.config.fm.m), scale=torch.tensor(self.config.fm.s))
        
        
    # Tool functions to set parameters, ema, optimizer, etc
    def restore_parameters(self, device: Optional[torch.device] = None) -> None:
        checkpoints_folder: str = self.checkpoints_folder
        prefix = ''
        if self.config.checkpoints_prefix:
            prefix = self.config.checkpoints_prefix
        ema_ckpt = torch.load(checkpoints_folder + '/' + prefix + '.pth')["ema"]
        self.ema.load_state_dict(ema_ckpt)

    def switch_to_ema(self) -> None:
        ema = self.ema
        model = self.model
        ema.store(model.parameters())
        ema.copy_to(model.parameters())
    
    def switch_back_from_ema(self) -> None:
        ema = self.ema
        model = self.model
        ema.restore(model.parameters())

    def set_optimizer(self) -> None:
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.optim.lr,
            weight_decay=self.config.optim.weight_decay,
            betas=(self.config.optim.beta_1, self.config.optim.beta_2),
            eps=self.config.optim.eps,
        )
        self.warmup = self.config.optim.linear_warmup
        self.grad_clip_norm = self.config.optim.grad_clip_norm
        self.optimizer = optimizer
    
    def set_scheduler(self) -> None:
        if self.config.scheduler.type == "cosine":
            self.scheduler = CosineLRScheduler(
                self.optimizer,
                t_initial=self.config.training.training_iters,
                lr_min=self.config.optim.min_lr,
                warmup_lr_init=self.config.optim.warmup_lr,
                warmup_t=self.config.optim.linear_warmup,
                cycle_limit=1,
                t_in_epochs=False,
            )
        elif self.config.scheduler.type == "anneal":
            lambda_scheduler = lambda i: (self.config.optim.max_lr-self.config.optim.warmup_lr)*i / self.config.optim.linear_warmup + self.config.optim.warmup_lr \
                                    if i < self.config.optim.linear_warmup else \
                                    (self.config.optim.min_lr+0.5*(self.config.optim.max_lr-self.config.optim.min_lr)* \
                                    (1.0+math.cos((i-self.config.optim.linear_warmup)/(self.config.training.training_iters \
                                    -self.config.optim.linear_warmup)*math.pi)))
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda_scheduler)

    def set_grad_scaler(self) -> None:
        self.grad_scaler = GradScaler()

    def set_sampler(self) -> None:
        self.sampler = OTPlanSampler(method=self.config.fm.ot_sampler_mode)

    def set_flow_matcher(self) -> None:
        self.flow_matcher = ConditionalFlowMatcher(sigma=self.config.fm.sigma)

    # Dataloaders
    def set_train_data_generator(self) -> None:
        if self.train_dataset is None:
            self.train_dataset = load_fasta_file(self.config.data.train_dataset_path)
        print("Train dataset length:", len(self.train_dataset))

        if self.config.ddp:
            num_tasks = dist.get_world_size()
            global_rank = dist.get_rank()

            sampler_train = torch.utils.data.DistributedSampler(
                self.train_dataset,
                num_replicas=num_tasks,
                rank=global_rank,
                shuffle=True,
            )
        else:
            sampler_train = None

        self.train_loader = DataLoader(
            self.train_dataset,
            sampler=sampler_train,
            batch_size=self.config.training.batch_size_per_gpu,
            num_workers=15,
            pin_memory=False,
        )

    def set_valid_data_generator(self) -> None:
        if self.valid_dataset is None:
            self.valid_dataset = load_fasta_file(self.config.data.test_dataset_path)
        print("Valid dataset length:", len(self.valid_dataset))

        if self.config.ddp:
            sampler_valid = torch.utils.data.distributed.DistributedSampler(
                self.valid_dataset,
                shuffle=False
            )
        else:
            sampler_valid = None

        self.valid_loader = DataLoader(
            self.valid_dataset,
            sampler=sampler_valid,
            batch_size=self.config.validation.batch_size // dist.get_world_size(),
            num_workers=15,
            pin_memory=False,
        )

    def set_train_reflow_data_generator(self) -> None:
        if self.train_dataset is None:
            self.train_dataset = ReflowDataset("train", self.config.fm.reflow_datapath, self.config.fm.reflow_datanum)
        print("Train dataset length:", len(self.train_dataset))

        if self.config.ddp:
            num_tasks = dist.get_world_size()
            global_rank = dist.get_rank()

            sampler_train = torch.utils.data.DistributedSampler(
                self.train_dataset,
                num_replicas=num_tasks,
                rank=global_rank,
                shuffle=True,
            )
        else:
            sampler_train = None

        self.train_loader = DataLoader(
            self.train_dataset,
            sampler=sampler_train,
            batch_size=self.config.training.batch_size_per_gpu,
            num_workers=15,
            pin_memory=False,
        )

    def set_valid_reflow_data_generator(self) -> None:
        if self.valid_dataset is None:
            self.valid_dataset = ReflowDataset("valid", self.config.fm.reflow_datapath, self.config.fm.reflow_datanum)
        print("Valid dataset length:", len(self.valid_dataset))

        if self.config.ddp:
            sampler_valid = torch.utils.data.distributed.DistributedSampler(
                self.valid_dataset,
                shuffle=False
            )
        else:
            sampler_valid = None

        self.valid_loader = DataLoader(
            self.valid_dataset,
            sampler=sampler_valid,
            batch_size=self.config.validation.batch_size // dist.get_world_size(),
            num_workers=15,
            pin_memory=False,
        )


    # logger
    def log_metric(self, metric_name: str, loader_name: str, value: Union[float, torch.Tensor, wandb.Image]):
        wandb.log({f'{metric_name}/{loader_name}': value}, step=self.step)



    # optimizer_step
    def optimizer_step(self, loss: torch.Tensor):
        self.optimizer.zero_grad()
        self.grad_scaler.scale(loss).backward()

        self.grad_scaler.unscale_(self.optimizer)

        if self.config.model.model_type != "cnn":
            grad_norm = torch.sqrt(sum([torch.sum(t.grad ** 2) for t in self.model.parameters()]))

            if self.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=self.grad_clip_norm
                )
        
            clipped_grad_norm = torch.sqrt(sum([torch.sum(t.grad ** 2) for t in self.model.parameters()]))
        if dist.get_rank() == 0:
            writer.add_scalar('Train/lr', self.optimizer.param_groups[0]['lr'], self.step)
        if self.config.wandb and dist.get_rank() == 0:
            self.log_metric('lr', 'train', self.optimizer.param_groups[0]['lr'])
        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()

        # Custom strategy
        scale = self.grad_scaler._scale.item()
        max_scale = 2 ** 30
        min_scale = 1
        scale = np.clip(scale, min_scale, max_scale)
        self.grad_scaler.update(new_scale=scale)

        self.ema.update(self.model.parameters())
        if self.config.scheduler.type == "cosine":
            self.scheduler.step_update(self.step)
        elif self.config.scheduler.type == "anneal":
            self.scheduler.step()
        if self.config.model.model_type != "cnn":
            return grad_norm, clipped_grad_norm
        else:
            return 0
    


    # training code
    def train(
            self,
            project_name: str = 'flow_matching',
            experiment_name: str = 'emb'
    ) -> None:
        self.step = 0
        self.set_optimizer()
        self.set_scheduler()
        self.set_grad_scaler()
        self.set_sampler()
        self.set_flow_matcher()
        if self.config.fm.reflow:
            self.set_valid_reflow_data_generator()
        else:
            self.set_valid_data_generator()
        self.ema = ExponentialMovingAverage(self.model.parameters(), self.config.model.ema_rate)

        if self.config.refresh.true:
            self.refresh_checkpoint()
            self.estimation()
            self.validate()

        if self.config.fm.reflow:
            load = torch.load(f'{self.config.fm.reflow_ckpt}', map_location="cpu")

            self.ema = ExponentialMovingAverage(self.model.parameters(), self.config.model.ema_rate)
            self.ema.load_state_dict(load["ema"])
            self.ema.cuda()
            self.switch_to_ema()

            print(f"Checkpoint refreshed {self.config.refresh.prefix}")

        self.train_range = trange(self.step + 1, self.config.training.training_iters + 1)
        self.train_range_iter = iter(self.train_range)

        while True:
            if self.config.fm.reflow:
                self.set_train_reflow_data_generator()
            else:
                self.set_train_data_generator()
            self.ddp_model.train()
            self.train_epoch()

            if self.step >= self.config.training.training_iters:
                break

        self.model.eval()
        self.save_checkpoint(last=True)
        self.switch_to_ema()
        writer.close()

    def train_epoch(self):
            for _, X in enumerate(self.train_loader):
                if self.step >= self.config.training.training_iters:
                    return
                _ = next(self.train_range_iter)

                loss_dict, stat_dict = self.train_step(X)

                if self.step % self.config.training.generate_freq == 0 and dist.get_rank() == 0:
                    print("Example Sequences: ", self.generate_text(5))

                if self.step % self.config.training.checkpoint_freq == 0:
                    self.save_checkpoint()

                if self.step % self.config.training.eval_freq == 0:
                    torch.cuda.empty_cache()
                    self.estimation()
                    self.validate()

                self.train_range.set_description(
                    f"loss: {loss_dict['loss'].item():0.4f}, "
                    f"grad_norm: {stat_dict['grad_norm'].item():0.4f}, "
                )
    
    def train_step(self, X):
        self.step += 1
        if self.config.fm.reflow:
            z0, z1 = X
            loss_dict, stat_dict = self.calc_loss(clean_x=z0.cuda(), X=z1.cuda())
        else:
            X = dict_to_cuda(X)
            with torch.no_grad():
                clean_X, tokenized_X = self.encoder_decoder.batch_encode(X)
                if self.config.use_compress:
                    tokens = tokenized_X["input_ids"]
                    mask = tokenized_X["attention_mask"]
                    clean_X = trim_or_pad_batch_first(clean_X, pad_to=self.config.data.max_sequence_len, pad_idx=0)
                    if mask.shape[1] != clean_X.shape[1]:
                        # pad with False
                        mask = trim_or_pad_batch_first(mask, clean_X.shape[1], pad_idx=0)
                        tokens = trim_or_pad_batch_first(tokens, clean_X.shape[1], pad_idx=1)
                    clean_X = clean_X.to(self.device)
                    mask = mask.to(self.device)
                    tokens = tokens.to(self.device)
                    clean_X = self.enc_normalizer.minmax_scaling(clean_X)
                    z_q, downsampled_mask = self.compresser.encode(x = clean_X, mask = mask.bool(), verbose=self.config.compress.verbose)
                    clean_X = z_q
                    tokenized_X = {"input_ids":tokens, "attention_mask":downsampled_mask}
            loss_dict, stat_dict = self.calc_loss(clean_x=clean_X, X=tokenized_X)

        stat_dict["grad_norm"], stat_dict["clipped_grad_norm"] = self.optimizer_step(loss_dict['total_loss'])

        if dist.get_rank() == 0:
            if self.step % 10 == 0:
                stat_dict["weight_norm"] = torch.sqrt(
                    sum([torch.sum(t.data ** 2) for t in self.model.parameters()]))
                if self.config.wandb:
                    for k, v in loss_dict.items():
                        self.log_metric(k, 'train', v.item())

                    for k, v in stat_dict.items():
                        self.log_metric(k, 'train', v.item())

        return loss_dict, stat_dict
    
    def calc_loss(
            self,
            clean_x, # clean_X: batch_size x seq_length x hidden_dim 128,50,320
            X=None, # tokenized_X: batch_size x seq_length 128,50
            eps: float = 1e-5,
    ) -> Dict[str, torch.Tensor]:
        if self.config.fm.reflow:
            mask = None
        elif self.config.fm.use_mask:
            mask = X["attention_mask"] # batch_size x seq_length 128,50
        else:
            mask = None

        batch_size = clean_x.size(0)

        if self.config.fm.reflow:
            x_0 = clean_x
            x_1 = X
        else:
            x_0 = torch.randn_like(clean_x)
            x_1 = clean_x

        if self.config.use_class:
            if self.config.use_compress:
                cls = torch.zeros((clean_x.shape[0], clean_x.shape[1], self.config.model.hidden_size), dtype=clean_x.dtype).cuda()
            else:
                cls = torch.zeros_like(clean_x, dtype=clean_x.dtype).cuda()
        else:
            cls = None

        t = torch.rand(batch_size).type_as(x_1)
        x_t = t.unsqueeze(1).unsqueeze(2)*x_1 + (1-t).unsqueeze(1).unsqueeze(2)*x_0
        u_t = x_1 - x_0

        with torch.autocast(device_type='cuda', dtype=torch.float32):
            v_t = self.ddp_model(
                    x_t=x_t, time_t=t,
                    attention_mask=mask,
                    cls=cls
                )

        weights = torch.ones_like(t, dtype=t.dtype)
        loss = torch.mean(weights.unsqueeze(1).unsqueeze(2)*(v_t - u_t) ** 2)
        loss_dict = {
            'total_loss': loss, 
            'loss': loss,
        }        
        clean_x_mean, clean_x_std, clean_x_norm = self.get_stat(clean_x, mask)
        x_0_mean, x_0_std, x_0_norm = self.get_stat(x_0, mask)
        stat_dict = {
            "clean_x_mean": clean_x_mean,
            "clean_x_std": clean_x_std,
            "clean_x_norm": clean_x_norm,
            "x_0_mean": x_0_mean,
            "x_0_std": x_0_std,
            "x_0_norm": x_0_norm,
        }
        return loss_dict, stat_dict

    def sample_time(self, batch_size: int, eps: float = 1e-5):
        return torch.cuda.FloatTensor(batch_size).uniform_() * (1 - eps) + eps
    
    def get_stat(self, z, mask):
        if mask is None:
            mask = torch.ones(
                (z.shape[0], z.shape[1]),
                device=f"cuda:{dist.get_rank()}",
                requires_grad=False,
                dtype=torch.int64,
            )
        mask_SEP_CLS = make_mask_wo_SEP_CLS(mask)
        mean = masked_mean(z, mask_SEP_CLS)
        std = masked_std(z, mask_SEP_CLS)
        norm = torch.sum(torch.norm(z, dim=2) * mask_SEP_CLS) / torch.sum(mask_SEP_CLS)
        return torch.mean(mean), torch.mean(std), norm


    # validation code
    def validate(self) -> None:
        prev_mode = self.ddp_model.training

        self.ddp_model.eval()
        self.switch_to_ema()

        valid_loss: Dict[str, torch.Tensor] = dict()
        valid_count = torch.Tensor([0.0])

        with torch.no_grad():
            for X in self.valid_loader:
                if self.config.fm.reflow:
                    z0, z1 = X
                    loss_dict, _ = self.calc_loss(clean_x=z0.cuda(), X=z1.cuda())
                    for k, v in loss_dict.items():
                        if k in valid_loss:
                            valid_loss[k] += v.item() * z0.size(0)
                        else:
                            valid_loss[k] = torch.Tensor([v.item() * z0.size(0)])
                    valid_count += z0.size(0)
                else:
                    X = dict_to_cuda(X)
                    clean_X, tokenized_X = self.encoder_decoder.batch_encode(X)
                    if self.config.use_compress:
                        tokens = tokenized_X["input_ids"]
                        mask = tokenized_X["attention_mask"]
                        clean_X = trim_or_pad_batch_first(clean_X, pad_to=self.config.data.max_sequence_len, pad_idx=0)
                        if mask.shape[1] != clean_X.shape[1]:
                            # pad with False
                            mask = trim_or_pad_batch_first(mask, clean_X.shape[1], pad_idx=0)
                            tokens = trim_or_pad_batch_first(tokens, clean_X.shape[1], pad_idx=1)
                        clean_X = clean_X.to(self.device)
                        mask = mask.to(self.device)
                        tokens = tokens.to(self.device)
                        clean_X = self.enc_normalizer.minmax_scaling(clean_X)
                        z_q, downsampled_mask = self.compresser.encode(x = clean_X, mask = mask.bool(), verbose=self.config.compress.verbose)
                        clean_X = z_q
                        tokenized_X = {"input_ids":tokens, "attention_mask":downsampled_mask}

                    loss_dict, _ = self.calc_loss(clean_x=clean_X, X=tokenized_X)
                    for k, v in loss_dict.items():
                        if k in valid_loss:
                            valid_loss[k] += v.item() * clean_X.size(0)
                        else:
                            valid_loss[k] = torch.Tensor([v.item() * clean_X.size(0)])
                    valid_count += clean_X.size(0)

        valid_count = reduce_tensor(valid_count.cuda())
        for k, v in valid_loss.items():
            valid_loss[k] = reduce_tensor(valid_loss[k].cuda())

        for k, v in valid_loss.items():
            valid_loss[k] = v / valid_count
        if self.config.wandb and dist.get_rank() == 0:
            for k, v in valid_loss.items():
                self.log_metric(k, 'valid_loader', v)

        self.switch_back_from_ema()
        self.ddp_model.train(prev_mode)



    def save_checkpoint(self, last: bool = False) -> None:
        if dist.get_rank() == 0:
            if not os.path.exists(self.checkpoints_folder):
                os.makedirs(self.checkpoints_folder)

            prefix = ''
            if self.config.checkpoints_prefix:
                prefix = self.config.checkpoints_prefix + '_'
            if last:
                prefix = prefix + 'last_'
            else:
                prefix = prefix + str(self.step) + '_'

            torch.save(
                {   
                    "model": self.model.state_dict(),
                    #"decoder": self.decoder.state_dict(), #no self.decoder exists
                    "ema": self.ema.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "scheduler": self.scheduler.state_dict(),
                    "step": self.step,
                },
                os.path.join(self.checkpoints_folder, prefix + ".pth")
            )
            print(f"Save model to: {os.path.join(self.checkpoints_folder, prefix + f'model.pth')}")



    def refresh_checkpoint(self):
        if not self.config.refresh.true:
            return
        load = torch.load(f'{self.config.refresh.prefix}', map_location="cpu")

        self.ema = ExponentialMovingAverage(self.model.parameters(), self.config.model.ema_rate)
        self.ema.load_state_dict(load["ema"])
        self.ema.cuda()
        self.switch_to_ema()

        if not self.config.refresh.use_pretrain:
            self.optimizer.load_state_dict(load["optimizer"])
            self.scheduler.load_state_dict(load["scheduler"])
            self.step = load["step"]
        print(f"Checkpoint refreshed {self.config.refresh.prefix}")
    


    def generate_text(self, batch_size):
        lens = self.length_sampler.sample(batch_size)
        attention_mask = torch.zeros((batch_size, self.config.data.max_sequence_len))
        for i in range(batch_size):
            for j in range(lens[i]):
                attention_mask[i, j] = 1

        attention_mask = attention_mask.cuda()

        with torch.no_grad():
            ### self.pred_embeddings Need to write with the new model
            if self.config.fm.use_mask:
                pred_embeddings = self.pred_embeddings(batch_size, attention_mask)
            else:
                pred_embeddings = self.pred_embeddings(batch_size)
            
            if self.config.fm.reflow and self.config.fm.reflow_generate_data:
                pred_embedding_z0, pred_embedding_z1 = pred_embeddings
                #print(pred_embeddings)
                output = pred_embeddings#(pred_embedding_z0, self.pred_logits(pred_embedding_z1, attention_mask))
                #print(output)
            else:
                output = self.pred_logits(pred_embeddings, attention_mask)
        return output

    def pred_logits(self, pred_embeddings, attention_mask):
        if self.config.use_compress:
            x_recons = self.compresser.decode(pred_embeddings, attention_mask, self.config.compress.verbose)
            x_recons_unscaled = self.enc_normalizer.undo_minmax_scaling(x_recons)
            output = self.encoder_decoder.batch_decode(x_recons_unscaled, attention_mask=attention_mask)
        else:
            output = self.encoder_decoder.batch_decode(pred_embeddings, attention_mask=attention_mask)
        return output
    
    @torch.no_grad()
    def pred_embeddings(
            self, batch_size: int,
            attention_mask=None,
    ) -> torch.Tensor:
        if self.config.use_compress:
            shape = (
                batch_size,
                self.config.data.max_sequence_len,
                self.config.model.compressed_hidden_size
            )
        else:
            shape = (
                batch_size,
                self.config.data.max_sequence_len,
                self.config.model.hidden_size
            )

        node = NeuralODE(
                torch_wrapper(self.ddp_model, attention_mask, self.config.model.model_type, self.config.use_compress, self.config.model.hidden_size), solver=self.config.fm.solver, sensitivity="adjoint", atol=1e-4, rtol=1e-4
            )
        
        with torch.no_grad():
            noise = torch.randn(shape) * torch.tensor(self.config.fm.sample_std)
            traj = node.trajectory(
                noise.to(self.device),
                t_span=torch.linspace(0, 1, self.config.sampling_step + 1).to(self.device),
                )
            if self.config.fm.reflow and self.config.fm.reflow_generate_data:
                pred_embeddings = (traj[0], traj[-1])
            else:
                pred_embeddings = traj[-1]

        return pred_embeddings

    @torch.no_grad()
    def estimation(self) -> None:
        self.model.eval()
        self.switch_to_ema()
        
        num_texts = int(self.config.validation.num_gen_texts / dist.get_world_size())
        if dist.get_rank() < self.config.validation.num_gen_texts % dist.get_world_size():
            num_texts += 1

        seed = self.config.seed + dist.get_rank()
        set_seed(seed)

        output = self.generate_text(batch_size=num_texts)

        result = [{"protein": p} for p in output]
        if self.config.ddp:
            result = gather_texts(result)

        if not self.config.ddp or dist.get_rank() == 0:
            texts_path = "./generated_seqs/" + self.suffix
            os.makedirs(texts_path, exist_ok=True)

            file_name = f"{texts_path}/{self.config.checkpoints_prefix}-{len(result)}.json"
            json.dump(result, open(file_name, "w"), indent=4)
            print(file_name)

            fid_value = calculate_fid_for_files(self.config.data.test_dataset_path, file_name)
            print(f"FID: {fid_value:0.5f}")
            with open("FID.txt","a") as f:
                f.write(f"FID: {fid_value:0.5f}"+"\n")

        if self.config.wandb and self.config.ddp and dist.get_rank() == 0:
            self.log_metric(metric_name="FID", loader_name="", value=fid_value)
        if dist.get_rank() == 0:
            writer.add_scalar('Valid/FID', fid_value, self.step)
            
        self.switch_back_from_ema()
        self.model.train()

    @torch.no_grad()
    def test(self, ckpt, test_num=None) -> None:

        self.ema = ExponentialMovingAverage(self.model.parameters(), self.config.model.ema_rate)
        ema_ckpt = torch.load(ckpt)["ema"]
        self.ema.load_state_dict(ema_ckpt)
        self.switch_to_ema()
        self.model.eval()
        print("Load model checkpoints DONE.")

        num_texts = int(self.config.validation.num_gen_texts / dist.get_world_size())
        if dist.get_rank() < self.config.validation.num_gen_texts % dist.get_world_size():
            num_texts += 1

        seed = self.config.seed + dist.get_rank()
        set_seed(seed)

        print("Start generating proteins...")
        torch.cuda.empty_cache()
        batch = 1024
        filtered_output = []
        z0_train = []
        z0_valid= []
        z1_train = []
        z1_valid= []
        cnt = 0
        for i in range(num_texts // batch):
            print("batch", i, "/", num_texts // batch)
            output = self.generate_text(batch)
            torch.cuda.empty_cache()
            if self.config.fm.reflow and self.config.fm.reflow_generate_data:
                z0s, z1s = output
                for z0, z1 in zip(z0s, z1s):
                    if cnt < 5270:
                        z0_valid.append(z0.cpu())
                        z1_valid.append(z1.cpu())
                    else:
                        z0_train.append(z0.cpu())
                        z1_train.append(z1.cpu())
                    cnt += 1
            else:
                for p in output:
                    if len(p) < 2: #4
                        continue
                    else:
                        new_p = ""
                        for aa in p:
                            if aa in "AFCUDNEQGHLIKOMPRSTVWY":
                                new_p += aa

                        filtered_output.append(new_p)
        result = [{"protein": p} for p in filtered_output]

        if self.config.ddp:
            result = gather_texts(result)
            if self.config.fm.reflow:
                z0_train = gather_texts(z0_train)
                z1_train = gather_texts(z1_train)
                z0_valid = gather_texts(z0_valid)
                z1_valid = gather_texts(z1_valid)
        
        if not self.config.ddp or dist.get_rank() == 0:
            if self.config.fm.reflow and self.config.fm.reflow_generate_data:
                texts_path = "" #TBD
            else:
                texts_path = "" #TBD
            os.makedirs(texts_path, exist_ok=True)

            if self.config.fm.reflow and self.config.fm.reflow_generate_data:
                z0_train_file_name = f"{texts_path}/z0_train"+str(test_num)+".npy"
                z1_train_file_name = f"{texts_path}/z1_train"+str(test_num)+".npy"
                z0_valid_file_name = f"{texts_path}/z0_valid.npy"
                z1_valid_file_name = f"{texts_path}/z1_valid.npy"
                print("Saving to files ...")
                np.save(z0_valid_file_name, np.array(z0_valid))
                np.save(z1_valid_file_name, np.array(z1_valid))
                print("Files saved.")
            else:
                file_name = f"{texts_path}/{self.config.checkpoints_prefix}-{len(result)}.json"
                json.dump(result, open(file_name, "w"), indent=4)
                print(file_name)

                fasta_file_name = f"{texts_path}/{self.config.checkpoints_prefix}-{len(result)}.fasta"
                with open(fasta_file_name, "w") as f:
                    cnt = 0
                    for i in result:
                        #if cnt < 1000: #1000
                            #print(i)
                        f.write(">Seq"+str(cnt)+"\n")
                        f.write(i["protein"]+"\n")
                        #else:
                        #    break
                        cnt += 1

                torch.cuda.empty_cache()
                fid_value = calculate_fid_for_files(self.config.data.test_dataset_path, file_name)
                print(f"FID: {fid_value:0.5f}")

    @torch.no_grad()
    def test_encoder_decoder(self) -> None:
        self.set_valid_data_generator()
        gathered_output = []
        for X in self.valid_loader:
            X = dict_to_cuda(X)
            pred_embeddings, tokenized_X = self.encoder_decoder.batch_encode(X)
            attention_mask = tokenized_X["attention_mask"]
            output = self.pred_logits(pred_embeddings, attention_mask)
            for seq in output:
                gathered_output.append(seq)

        result = [{"protein": p} for p in gathered_output]

        if self.config.ddp:
            result = gather_texts(result)

        if not self.config.ddp or dist.get_rank() == 0:
            texts_path = "./generated_seqs/test"
            os.makedirs(texts_path, exist_ok=True)

            file_name = f"{texts_path}/{self.config.checkpoints_prefix}-{len(result)}.json"
            json.dump(result, open(file_name, "w"), indent=4)
            fid_value = calculate_fid_for_files(self.config.data.test_dataset_path, file_name)
            print(f"FID: {fid_value:0.5f}")

