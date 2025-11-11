import typing as T
import numpy as np
import torch
from torch import nn

from compressers.modules import HourglassDecoder, HourglassEncoder, VectorQuantizer, FiniteScalarQuantizer

def trim_or_pad_batch_first(tensor: torch.Tensor, pad_to: int, pad_idx: int = 0):
    """Trim or pad a tensor with shape (B, L, ...) to a given length."""
    N, L = tensor.shape[0], tensor.shape[1]
    if L >= pad_to:
        tensor = tensor[:, :pad_to, ...]
    elif L < pad_to:
        padding = torch.full(
            size=(N, pad_to - L, *tensor.shape[2:]),
            fill_value=pad_idx,
            dtype=tensor.dtype,
            device=tensor.device,
        )
        tensor = torch.concat((tensor, padding), dim=1)
    return tensor

class HourglassProteinCompressionTransformer(nn.Module):
    def __init__(self,
                 dim, depth, downproj_factor, shorten_factor, attn_resampling, updown_sample_type, heads, dim_head, causal, norm_out,
                 use_quantizer, n_e, e_dim, vq_beta, enforce_single_codebook_per_position, fsq_levels):
        super().__init__()

        # set up quantizer
        assert use_quantizer in ["vq", "fsq", "tanh"]
        self.quantize_scheme = use_quantizer
        print(f"Using {use_quantizer} layer at bottleneck...")
        self.pre_quant_proj = None
        self.post_quant_proj = None
        self.quantizer = None
        
        # Set up encoder/decoders
        self.enc = HourglassEncoder(
            dim=dim,
            depth=depth,
            shorten_factor=shorten_factor,
            downproj_factor=downproj_factor,
            attn_resampling=attn_resampling,
            updown_sample_type=updown_sample_type,
            heads=heads,
            dim_head=dim_head,
            causal=causal,
            norm_out=norm_out,
        ).cuda()

        self.dec = HourglassDecoder(
            dim=dim // downproj_factor,
            depth=depth,
            elongate_factor=shorten_factor,
            upproj_factor=downproj_factor,
            attn_resampling=True,
            updown_sample_type=updown_sample_type,
        ).cuda()
        
        self.parameters = None

        # other misc settings
        self.dim = dim
        self.z_q_dim = self.dim // np.prod(self.dim)
        self.n_e = n_e
        self.downproj_factor = downproj_factor 
        self.e_dim = e_dim
        self.vq_beta = vq_beta
        self.enforce_single_codebook_per_position = enforce_single_codebook_per_position
        self.fsq_levels = fsq_levels

    def set_up_quantizer(self, dim, downproj_factor, n_e, e_dim, vq_beta, enforce_single_codebook_per_position, fsq_levels, implicit_codebook):
        if self.quantize_scheme == "vq":
            self.quantizer = VectorQuantizer(n_e, e_dim, vq_beta).cuda()

            # if this is enforced, then we'll project down the channel dimension to make sure that the
            # output of the encoder has the same dimension as the embedding codebook.
            # otherwise, the excess channel dimensions will be tiled up lengthwise,
            # which combinatorially increases the size of the codebook. The latter will
            # probably lead to better results, but is not the convention and may lead to
            # an excessively large codebook for purposes such as training an AR model downstream.
            if enforce_single_codebook_per_position and (
                dim / downproj_factor != e_dim
            ):
                self.pre_quant_proj = torch.nn.Linear(dim // downproj_factor, e_dim).cuda()
                self.post_quant_proj = torch.nn.Linear(e_dim, dim // downproj_factor).cuda()

        elif self.quantize_scheme == "fsq":
            if not len(fsq_levels) == (dim / downproj_factor):
                # similarly, project down to the length of the FSQ vectors.
                # unlike with VQ-VAE, the convention with FSQ *is* to combinatorially incraese the size of codebook
                self.pre_quant_proj = torch.nn.Linear(
                    dim // downproj_factor, len(fsq_levels)
                ).cuda()
                self.post_quant_proj = torch.nn.Linear(
                    len(fsq_levels), dim // downproj_factor
                ).cuda()
            self.fsq_levels = fsq_levels
            self.quantizer = FiniteScalarQuantizer(fsq_levels).cuda()
        else:
            # self.quantize_scheme in [None, "tanh"]
            self.quantizer = None

    def encode(self, x, mask=None, verbose=False, *args, **kwargs):
        if mask is None:
            mask = torch.ones((x.shape[0], x.shape[1])).to(x.device)
            mask = mask.bool()

        # ensure that input length is a multiple of the shorten factor
        s = self.enc.shorten_factor
        extra = x.shape[1] % s
        if extra != 0:
            needed = s - extra
            x = trim_or_pad_batch_first(x, pad_to=x.shape[1] + needed, pad_idx=0)

        # In any case where the mask and token generated from sequence strings don't match latent, make it match
        if mask.shape[1] != x.shape[1]:
            # pad with False
            mask = trim_or_pad_batch_first(mask, x.shape[1], pad_idx=0)

        # encode and possibly downsample
        log_dict = {}
        z_e, downsampled_mask = self.enc(x, mask, verbose)

        # if encoder output dimensions does not match quantization inputs, project down
        if self.pre_quant_proj is not None:
            z_e = self.pre_quant_proj(z_e)

        ##################
        # Quantize
        ##################

        # VQ-VAE

        if self.quantize_scheme == "vq":
            quant_out = self.quantizer(z_e, verbose)
            z_q = quant_out["z_q"]
            vq_loss = quant_out["loss"]
            log_dict["vq_loss"] = quant_out["loss"]
            log_dict["vq_perplexity"] = quant_out["perplexity"]
            compressed_representation = quant_out[
                "min_encoding_indices"
            ].detach()  # .cpu().numpy()

        # FSQ

        elif self.quantize_scheme == "fsq":
            z_q = self.quantizer.quantize(z_e)
            compressed_representation = self.quantizer.codes_to_indexes(
                z_q
            ).detach()  # .cpu().numpy()

        # Continuous (no quantization) with a tanh bottleneck

        elif self.quantize_scheme == "tanh":
            z_e = z_e.to(torch.promote_types(z_e.dtype, torch.float32))
            z_q = torch.tanh(z_e)
        
        else:
            raise NotImplementedError

        #if infer_only:
        #    compressed_representation = z_q.detach()  # .cpu().numpy()
        #    downsampled_mask = downsampled_mask.detach()  # .cpu().numpy()
        #    return compressed_representation, downsampled_mask
        #else:
        #    return z_q, downsampled_mask
        return z_q, downsampled_mask

    def decode(self, z_q, downsampled_mask=None, verbose=False):
        if self.post_quant_proj is not None:
            z_q = self.post_quant_proj(z_q)

        x_recons = self.dec(z_q, downsampled_mask, verbose)
        return x_recons
    
    def state_dict(self):
        state = super().state_dict()
        state = {k: v for k, v in state.items() if "esm" not in k}
        return state
    
    def from_pretrained(self, checkpoint_path):
        state = torch.load(checkpoint_path)
        self.enc.load_state_dict(state["encoder"])
        self.dec.load_state_dict(state["decoder"])
        if self.quantize_scheme == "fsq" or self.quantize_scheme == "vq":
            self.fsq_levels=state["quantizer"][1]
            self.implicit_codebook=state["quantizer"][2]
            self.set_up_quantizer(self.dim, self.downproj_factor, self.n_e, self.e_dim, self.vq_beta, self.enforce_single_codebook_per_position, self.fsq_levels, self.implicit_codebook)
            self.quantizer.load_state_dict(state["quantizer"][0])
            self.pre_quant_proj.load_state_dict(state["pre_quant_proj"])
            self.post_quant_proj.load_state_dict(state["post_quant_proj"])
            
        
