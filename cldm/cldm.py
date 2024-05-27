import einops
import torch
import torch as th
import torch.nn as nn
import numpy as np
from collections import deque

from einops import rearrange, repeat
from torchvision.utils import make_grid
from ldm.modules.attention import SpatialTransformer
from ldm.modules.diffusionmodules.openaimodel import (
        UNetModel, 
        TimestepEmbedSequential, 
        ResBlock, 
        Upsample, 
        Downsample, 
        AttentionBlock, 
        TimestepBlock,
        normalization
)

from ldm.models.diffusion.ddpm import LatentDiffusion,disabled_train, __conditioning_keys__
from ldm.util import log_txt_as_img, exists, instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.modules.diffusionmodules.util import extract_into_tensor

from ldm.modules.diffusionmodules.util import (
    checkpoint,
    conv_nd,
    linear,
    zero_module,
    normalization,
    timestep_embedding,
)
    

class ResZeroBlock(nn.Module):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """
    def __init__(
        self,
        channels,
        dropout,
        out_channels=None,
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
    ):
        super().__init__()
        self.channels = channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_checkpoint = use_checkpoint

        # self.zero_param = zero_module(torch.nn.Parameter(torch.tensor(1.)[..., None, None, None]))
        self.out_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.
        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward, (x,), self.parameters(), self.use_checkpoint
        )

    def _forward(self, x):
        h = self.out_layers(x)
        return self.skip_connection(x) + h


class BPS(nn.Module):
    def __init__(
        self,
        unshuffle_dim=8,
        in_channels=64,
        out_channels=4,
        model_channels=320,
        num_res_blocks=2,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        use_checkpoint=False,
        use_fp16=False,
        use_scale_shift_norm=True,
        is_controlnet = False,
        res_weights = [16,16,1,8,8,1,4,4,1,2,2,1,16],
        res_w_requires_grad = False,
    ):
        super().__init__()

        self.is_controlnet = is_controlnet
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        if isinstance(num_res_blocks, int):
            self.num_res_blocks = len(channel_mult) * [num_res_blocks]
        else:
            if len(num_res_blocks) != len(channel_mult):
                raise ValueError("provide num_res_blocks either as an int (globally constant) or "
                                 "as a list/tuple (per-level) with the same length as channel_mult")
            self.num_res_blocks = num_res_blocks

        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.dtype = th.float16 if use_fp16 else th.float32

        time_embed_dim = model_channels * 4

        self.unshuffle = nn.PixelUnshuffle(unshuffle_dim) # 1024/16=64 512/8=64 channel = 8*8*3

        self.res_weights = nn.ParameterList(
            [nn.Parameter(torch.tensor(i,dtype=self.dtype), 
                          requires_grad=res_w_requires_grad) for i in [1]+res_weights]
            )
        self.input_blocks = nn.ModuleList(
            [
                conv_nd(dims, in_channels, model_channels, 3, padding=1)
            ]
        )
        self.zero_blocks = nn.ModuleList([
            zero_module(conv_nd(dims, model_channels, model_channels, 3, padding=1))
        ])

        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for nr in range(self.num_res_blocks[level]):
                self.input_blocks.append(
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                )
                ch = mult * model_channels
                self.zero_blocks.append(
                    zero_module(conv_nd(dims, ch, ch, 3, padding=1))
                )
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                        Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                )
                self.zero_blocks.append(
                    zero_module(conv_nd(dims, out_ch, out_ch, 3, padding=1))
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self.zero_blocks.append(zero_module(conv_nd(dims, ch, ch, 3, padding=1)))
        
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
     
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(self.num_res_blocks[level] + 1):
                ich = input_block_chans.pop()
                layers = nn.ModuleList(TimestepEmbedSequential(
                    ResZeroBlock(
                        ch + ich,
                        # time_embed_dim,
                        dropout,
                        out_channels=model_channels * mult,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        # use_scale_shift_norm=use_scale_shift_norm,
                    ),
                    zero_module(conv_nd(dims, model_channels * mult, model_channels * mult, 3, padding=1))
                )
                )
                ch = model_channels * mult

                if level and i == self.num_res_blocks[level]:
                    out_ch = ch
                    layers.append(
                        Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(layers)
                self._feature_size += ch

    def forward(self, x, time_emb=None):
        # unshuffle
        x = self.unshuffle(x)
        emb = time_emb

        hs = []
        outs = deque()
        add_res =  []

        h = x.type(self.dtype)
        for i, module in enumerate(self.input_blocks):
            if isinstance(module, TimestepBlock):
                h = module(h, emb)
                zero_h = self.zero_blocks[i](h)
                outs.append(self.res_weights[i]*zero_h)
            else:
                h = module(h)
                zero_h = self.zero_blocks[i](h)
                outs.append(None)
            hs.append(h)
            if self.is_controlnet:
                add_res.append(zero_h)
            

        h = self.middle_block(h, emb)
        outs.append(self.res_weights[-1]*self.zero_blocks[-1](h))

        for modules in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = modules[0](h)
            outs.append(modules[1](h))  # have zero_block
            if len(modules) > 2:
                h = modules[2](h)

        return outs, add_res


class BPS_forstyle(BPS):
    def __init__(self, unshuffle_dim_add=8, **kwargs):
        super().__init__(**kwargs)
        self.unshuffle1 = nn.PixelUnshuffle(unshuffle_dim_add) # 1024/16=64 512/8=64 channel = 8*8*3
    def forward(self, x, time_emb=None):
        # unshuffle
        x = torch.cat([self.unshuffle(x[:,:3,:,:]), self.unshuffle1(x[:,3,:,:].unsqueeze(1))],dim=1)
        emb = time_emb

        hs = []
        outs = deque()
        add_res =  []

        h = x.type(self.dtype)
        for i, module in enumerate(self.input_blocks):
            if isinstance(module, TimestepBlock):
                h = module(h, emb)
                zero_h = self.zero_blocks[i](h)
                outs.append(self.res_weights[i]*zero_h)
            else:
                h = module(h)
                zero_h = self.zero_blocks[i](h)
                outs.append(None)
            hs.append(h)
            if self.is_controlnet:
                add_res.append(zero_h)
            

        h = self.middle_block(h, emb)
        outs.append(self.res_weights[-1]*self.zero_blocks[-1](h))

        for modules in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = modules[0](h)
            outs.append(modules[1](h))  # have zero_block
            if len(modules) > 2:
                h = modules[2](h)

        return outs, add_res


class BPS_XL(nn.Module):
    def __init__(
        self,
        in_channels=64,
        out_channels=4,
        model_channels=320,
        num_res_blocks=2,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        attention_resolutions = (4,2,1),
        num_heads=-1,
        num_head_channels=-1,
        num_attention_blocks = None,
        use_new_attention_order=False,
        dims=2,
        conv_resample=True,
        use_checkpoint=False,
        use_fp16=False,
        use_scale_shift_norm=True,
        is_controlnet = False,
        res_weights = [32,32,1,16,16,1,8,8,1,4,4,1,32],
        res_w_requires_grad = False,
    ):
        super().__init__()

        self.is_controlnet = is_controlnet
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        if isinstance(num_res_blocks, int):
            self.num_res_blocks = len(channel_mult) * [num_res_blocks]
        else:
            if len(num_res_blocks) != len(channel_mult):
                raise ValueError("provide num_res_blocks either as an int (globally constant) or "
                                 "as a list/tuple (per-level) with the same length as channel_mult")
            self.num_res_blocks = num_res_blocks
        if num_attention_blocks is not None:
            assert len(num_attention_blocks) == len(self.num_res_blocks)
            assert all(map(lambda i: self.num_res_blocks[i] >= num_attention_blocks[i], range(len(num_attention_blocks))))
            print(f"Constructor of UNetModel received num_attention_blocks={num_attention_blocks}. "
                  f"This option has LESS priority than attention_resolutions {attention_resolutions}, "
                  f"i.e., in cases where num_attention_blocks[i] > 0 but 2**i not in attention_resolutions, "
                  f"attention will still not be set.")

        self.attention_resolutions = attention_resolutions
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.dtype = th.float16 if use_fp16 else th.float32

        time_embed_dim = model_channels * 4

        self.unshuffle = nn.PixelUnshuffle(8) # 1024/16=64 512/8=64 channel = 8*8*3

        self.res_weights = nn.ParameterList(
            [nn.Parameter(torch.tensor(i,dtype=self.dtype), 
                          requires_grad=res_w_requires_grad) for i in [1]+res_weights]
            )
        self.input_blocks = nn.ModuleList(
            [
                conv_nd(dims, in_channels, model_channels, 3, padding=1)
            ]
        )
        self.zero_blocks = nn.ModuleList([
            nn.Identity() # zero_module(conv_nd(dims, model_channels, model_channels, 3, padding=1))
        ])

        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for nr in range(self.num_res_blocks[level]):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                if not exists(num_attention_blocks) or nr < num_attention_blocks[level]:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=dim_head,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self.zero_blocks.append(
                    zero_module(conv_nd(dims, ch, ch, 3, padding=1))
                )
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                        Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                )
                self.zero_blocks.append(
                    nn.Identity() # zero_module(conv_nd(dims, out_ch, out_ch, 3, padding=1))
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=dim_head,
                use_new_attention_order=use_new_attention_order,
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self.zero_blocks.append(zero_module(conv_nd(dims, ch, ch, 3, padding=1)))
        
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
     
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(self.num_res_blocks[level] + 1):
                ich = input_block_chans.pop()
                layers = nn.ModuleList(TimestepEmbedSequential(
                    ResZeroBlock(
                        ch + ich,
                        # time_embed_dim,
                        dropout,
                        out_channels=model_channels * mult,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        # use_scale_shift_norm=use_scale_shift_norm,
                    ),
                    zero_module(conv_nd(dims, model_channels * mult, model_channels * mult, 3, padding=1))
                )
                )
                ch = model_channels * mult

                if level and i == self.num_res_blocks[level]:
                    out_ch = ch
                    layers.append(
                        Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(layers)
                self._feature_size += ch

    def forward(self, x, time_emb=None):
        # unshuffle
        x = self.unshuffle(x)
        emb = time_emb

        hs = []
        outs = deque()


        h = x.type(self.dtype)
        for i, module in enumerate(self.input_blocks):
            if isinstance(module, TimestepBlock):
                h = module(h, emb)
                zero_h = self.zero_blocks[i](h)
                outs.append(self.res_weights[i]*zero_h)
            else:
                h = module(h)
                # zero_h = self.zero_blocks[i](h)
                outs.append(None)
            hs.append(h)

        h = self.middle_block(h, emb)
        outs.append(self.res_weights[-1]*self.zero_blocks[-1](h))

        for modules in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = modules[0](h)
            outs.append(modules[1](h))  # have zero_block
            if len(modules) > 2:
                h = modules[2](h)

        return outs, []


class ControlledUnetModel(UNetModel):
    def __init__(self, control_stage_config, is_controlnet=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.control_model = instantiate_from_config(control_stage_config)
        self.is_controlnet = is_controlnet
        self.control_model.is_controlnet = is_controlnet
    def forward(self, x, timesteps=None, context=None, hint=None, **kwargs):
        hs = []
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)
        h = x.type(self.dtype)
    
        add_res, controls = self.control_model(hint, emb)

        for module in self.input_blocks:
            cond = add_res.popleft()
            h = module(h, emb, context, additional_res=cond)
            hs.append(h)

        h = self.middle_block(h, emb, context, additional_res=add_res.popleft())

        for i, module in enumerate(self.output_blocks):
            if self.is_controlnet:
                h = torch.cat([h, hs.pop()+ controls.pop()], dim=1) 
            else:
                h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb, context, additional_res=add_res.popleft()) # additional_res=add_res.popleft()

        h = h.type(x.dtype)
        return self.out(h)


class SteeringLDM_layout(LatentDiffusion):

    def __init__(self, control_key, only_mid_control, global_average_pooling=False, pre_train_path="./models/v1-5-pruned.ckpt", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.control_key = control_key
        self.only_mid_control = only_mid_control
        self.control_scales = [1.0] * 13
        self.global_average_pooling = global_average_pooling
        self. pre_train_path =  pre_train_path
    
    @torch.no_grad()
    def get_input(self, batch, k, bs=None, *args, **kwargs):
        x, c = super().get_input(batch, self.first_stage_key, *args, **kwargs)
        control = batch[self.control_key]
        if bs is not None:
            control = control[:bs]
        control = control.to(self.device)
        control = einops.rearrange(control, 'b h w c -> b c h w')
        control = control.to(memory_format=torch.contiguous_format).float()
        return x, dict(c_crossattn=[c], c_concat=[control])

    def apply_model(self, x_noisy, t, cond, *args, **kwargs):
        assert isinstance(cond, dict)
        diffusion_model = self.model.diffusion_model

        cond_txt = torch.cat(cond['c_crossattn'], 1)
        eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, hint=torch.cat(cond['c_concat'], 1), only_mid_control=self.only_mid_control)

        return eps

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.model.diffusion_model.control_model.parameters())
        if not self.sd_locked:
            params += list(self.model.diffusion_model.output_blocks.parameters())
            params += list(self.model.diffusion_model.out.parameters())
        opt = torch.optim.AdamW(params, lr=lr)
        return opt

    def low_vram_shift(self, is_diffusing):
        if is_diffusing:
            self.model = self.model.cuda()
            self.first_stage_model = self.first_stage_model.cpu()
            self.cond_stage_model = self.cond_stage_model.cpu()
        else:
            self.model = self.model.cpu()
            self.first_stage_model = self.first_stage_model.cuda()
            self.cond_stage_model = self.cond_stage_model.cuda()
    
    @torch.no_grad()
    def get_unconditional_conditioning(self, N):
        return self.get_learned_conditioning([""] * N)

    @torch.no_grad()
    def log_images(self, batch, N=4, n_row=2, sample=False, ddim_steps=50, ddim_eta=0.0, return_keys=None,
                   quantize_denoised=True, inpaint=True, plot_denoise_rows=False, plot_progressive_rows=True,
                   plot_diffusion_rows=False, unconditional_guidance_scale=9.0, unconditional_guidance_label=None,
                   use_ema_scope=True,
                   **kwargs):
        use_ddim = ddim_steps is not None

        log = dict()
        z, c = self.get_input(batch, self.first_stage_key, bs=N)
        c_cat, c = c["c_concat"][0][:N], c["c_crossattn"][0][:N]
        N = min(z.shape[0], N)
        n_row = min(z.shape[0], n_row)
        log["reconstruction"] = self.decode_first_stage(z)
        log["control"] = c_cat * 2.0 - 1.0
        log["conditioning"] = log_txt_as_img((512, 512), batch[self.cond_stage_key], size=16)

        if plot_diffusion_rows:
            # get diffusion row
            diffusion_row = list()
            z_start = z[:n_row]
            for t in range(self.num_timesteps):
                if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                    t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
                    t = t.to(self.device).long()
                    noise = torch.randn_like(z_start)
                    z_noisy = self.q_sample(x_start=z_start, t=t, noise=noise)
                    diffusion_row.append(self.decode_first_stage(z_noisy))

            diffusion_row = torch.stack(diffusion_row) 
            diffusion_grid = rearrange(diffusion_row, 'n b c h w -> b n c h w')
            diffusion_grid = rearrange(diffusion_grid, 'b n c h w -> (b n) c h w')
            diffusion_grid = make_grid(diffusion_grid, nrow=diffusion_row.shape[0])
            log["diffusion_row"] = diffusion_grid

        if sample:
            # get denoise row
            samples, z_denoise_row = self.sample_log(cond={"c_concat": [c_cat], "c_crossattn": [c]},
                                                     batch_size=N, ddim=use_ddim,
                                                     ddim_steps=ddim_steps, eta=ddim_eta)
            x_samples = self.decode_first_stage(samples)
            log["samples"] = x_samples
            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                log["denoise_row"] = denoise_grid

        if unconditional_guidance_scale > 1.0:
            uc_cross = self.get_unconditional_conditioning(N)

            uc_cat = c_cat  # torch.zeros_like(c_cat)
            uc_full = {"c_concat": [uc_cat], "c_crossattn": [uc_cross]}
            samples_cfg, _ = self.sample_log(cond={"c_concat": [c_cat], "c_crossattn": [c]},
                                             batch_size=N, ddim=use_ddim,
                                             ddim_steps=ddim_steps, eta=ddim_eta,
                                             unconditional_guidance_scale=unconditional_guidance_scale,
                                             unconditional_conditioning=uc_full,
                                             )
            x_samples_cfg = self.decode_first_stage(samples_cfg)
            log[f"samples_cfg_scale_{unconditional_guidance_scale:.2f}"] = x_samples_cfg

        return log

    @torch.no_grad()
    def sample_log(self, cond, batch_size, ddim, ddim_steps, **kwargs):
        ddim_sampler = DDIMSampler(self)
        b, c, h, w = cond["c_concat"][0].shape
        shape = (self.channels, h // 8, w // 8)
        samples, intermediates = ddim_sampler.sample(ddim_steps, batch_size, shape, cond, verbose=False, **kwargs)
        return samples, intermediates
    
    # def on_load_checkpoint(self, checkpoint):
    #     # self.model.diffusion_model.control_model.load_state_dict(checkpoint["controlnet"])
    #     params = checkpoint["controlnet"]
    #     checkpoint["state_dict"] = load_state_dict(self.pre_train_path, location='cpu')
    #     prefix = "model.diffusion_model.control_model"
    #     for name in checkpoint["controlnet"].keys():
    #         checkpoint["state_dict"].update({prefix+name:params[name]})

    # def on_save_checkpoint(self, checkpoint):
    #     checkpoint["controlnet"] = self.model.diffusion_model.control_model.state_dict()
    #     checkpoint.pop("state_dict")


class SteeringLDM(SteeringLDM_layout):
    @torch.no_grad()
    def log_images(self, batch, N=4, n_row=2, sample=False, ddim_steps=50, ddim_eta=0.0, return_keys=None,
                   quantize_denoised=True, inpaint=True, plot_denoise_rows=False, plot_progressive_rows=True,
                   plot_diffusion_rows=False, unconditional_guidance_scale=9.0, unconditional_guidance_label=None,
                   use_ema_scope=True,
                   **kwargs):
        use_ddim = ddim_steps is not None

        log = dict()
        z, c = self.get_input(batch, self.first_stage_key, bs=N)
        c_cat, c = c["c_concat"][0][:N], c["c_crossattn"][0][:N]
        N = min(z.shape[0], N)
        n_row = min(z.shape[0], n_row)
        log["reconstruction"] = self.decode_first_stage(z)
        log["control"] = batch['c_img'][:N] * 2.0 - 1.0
        log["conditioning"] = log_txt_as_img((512, 512), batch[self.cond_stage_key], size=16)

        if plot_diffusion_rows:
            # get diffusion row
            diffusion_row = list()
            z_start = z[:n_row]
            for t in range(self.num_timesteps):
                if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                    t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
                    t = t.to(self.device).long()
                    noise = torch.randn_like(z_start)
                    z_noisy = self.q_sample(x_start=z_start, t=t, noise=noise)
                    diffusion_row.append(self.decode_first_stage(z_noisy))

            diffusion_row = torch.stack(diffusion_row)  # n_log_step, n_row, C, H, W
            diffusion_grid = rearrange(diffusion_row, 'n b c h w -> b n c h w')
            diffusion_grid = rearrange(diffusion_grid, 'b n c h w -> (b n) c h w')
            diffusion_grid = make_grid(diffusion_grid, nrow=diffusion_row.shape[0])
            log["diffusion_row"] = diffusion_grid

        if sample:
            # get denoise row
            samples, z_denoise_row = self.sample_log(cond={"c_concat": [c_cat], "c_crossattn": [c]},
                                                     batch_size=N, ddim=use_ddim,
                                                     ddim_steps=ddim_steps, eta=ddim_eta)
            x_samples = self.decode_first_stage(samples)
            log["samples"] = x_samples
            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                log["denoise_row"] = denoise_grid

        if unconditional_guidance_scale > 1.0:
            uc_cross = self.get_unconditional_conditioning(N)

            uc_cat = c_cat  # torch.zeros_like(c_cat)
            uc_full = {"c_concat": [uc_cat], "c_crossattn": [uc_cross]}
            samples_cfg, _ = self.sample_log(cond={"c_concat": [c_cat], "c_crossattn": [c]},
                                             batch_size=N, ddim=use_ddim,
                                             ddim_steps=ddim_steps, eta=ddim_eta,
                                             unconditional_guidance_scale=unconditional_guidance_scale,
                                             unconditional_conditioning=uc_full,
                                             )
            x_samples_cfg = self.decode_first_stage(samples_cfg)
            log[f"samples_cfg_scale_{unconditional_guidance_scale:.2f}"] = x_samples_cfg

        return log

    @torch.no_grad()
    def sample_log(self, cond, batch_size, ddim, ddim_steps, **kwargs):
        ddim_sampler = DDIMSampler(self)
        b, c, h, w = cond["c_concat"][0].shape
        shape = (self.channels, 64, 64)
        samples, intermediates = ddim_sampler.sample(ddim_steps, batch_size, shape, cond, verbose=False, **kwargs)
        return samples, intermediates
