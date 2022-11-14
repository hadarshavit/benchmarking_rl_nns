"""
This file defines all the neural network architectures available to use.
"""
from functools import partial
from math import sqrt

import torch
from torch import nn as nn, Tensor
from torch.nn import init
import torch.nn.functional as F
import timm


def conv_output_size(in_size, kernel_size, stride, padding):
    return (in_size - kernel_size + 2*padding) // stride + 1


class FactorizedNoisyLinear(nn.Module):
    """ The factorized Gaussian noise layer for noisy-nets dqn. """
    def __init__(self, in_features: int, out_features: int, sigma_0: float) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_0 = sigma_0

        # weight: w = \mu^w + \sigma^w . \epsilon^w
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))

        # bias: b = \mu^b + \sigma^b . \epsilon^b
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise()

    @torch.no_grad()
    def reset_parameters(self) -> None:
        # initialization is similar to Kaiming uniform (He. initialization) with fan_mode=fan_in
        scale = 1 / sqrt(self.in_features)

        init.uniform_(self.weight_mu, -scale, scale)
        init.uniform_(self.bias_mu, -scale, scale)

        init.constant_(self.weight_sigma, self.sigma_0 * scale)
        init.constant_(self.bias_sigma, self.sigma_0 * scale)

    @torch.no_grad()
    def _get_noise(self, size: int) -> Tensor:
        noise = torch.randn(size, device=self.weight_mu.device)
        # f(x) = sgn(x)sqrt(|x|)
        return noise.sign().mul_(noise.abs().sqrt_())

    @torch.no_grad()
    def reset_noise(self) -> None:
        # like in eq 10 and 11 of the paper
        epsilon_in = self._get_noise(self.in_features)
        epsilon_out = self._get_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    @torch.no_grad()
    def disable_noise(self) -> None:
        self.weight_epsilon[:] = 0
        self.bias_epsilon[:] = 0

    def forward(self, input: Tensor) -> Tensor:
        # y = wx + d, where
        # w = \mu^w + \sigma^w * \epsilon^w
        # b = \mu^b + \sigma^b * \epsilon^b
        return F.linear(input,
                        self.weight_mu + self.weight_sigma*self.weight_epsilon,
                        self.bias_mu + self.bias_sigma*self.bias_epsilon)

class Dueling(nn.Module):
    """ The dueling branch used in all nets that use dueling-dqn. """
    def __init__(self, value_branch, advantage_branch):
        super().__init__()
        self.flatten = nn.Flatten()
        self.value_branch = value_branch
        self.advantage_branch = advantage_branch

    def forward(self, x, advantages_only=False):
        x = self.flatten(x)
        advantages = self.advantage_branch(x)
        if advantages_only:
            return advantages

        value = self.value_branch(x)
        return value + (advantages - torch.mean(advantages, dim=1, keepdim=True))


class DuelingAlt(nn.Module):
    """ The dueling branch used in all nets that use dueling-dqn. """
    def __init__(self, l1, l2):
        super().__init__()
        self.main = nn.Sequential(
            nn.Flatten(),
            l1,
            nn.ReLU(),
            l2
        )

    def forward(self, x, advantages_only=False):
        res = self.main(x)
        advantages = res[:, 1:]
        value = res[:, 0:1]
        return value + (advantages - torch.mean(advantages, dim=1, keepdim=True))

class NatureCNN(nn.Module):
    """
    This is the CNN that was introduced in Mnih et al. (2013) and then used in a lot of later work such as
    Mnih et al. (2015) and the Rainbow paper. This implementation only works with a frame resolution of 84x84.
    """
    def __init__(self, depth, actions, linear_layer):
        super().__init__()

        self.main = nn.Sequential(
            nn.Conv2d(in_channels=depth, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            linear_layer(3136, 512),
            nn.ReLU(),
            linear_layer(512, actions),
        )

    def forward(self, x, advantages_only=None):
        return self.main(x)


class DuelingNatureCNN(nn.Module):
    """
    Implementation of the dueling architecture introduced in Wang et al. (2015).
    This implementation only works with a frame resolution of 84x84.
    """
    def __init__(self, depth, actions, linear_layer):
        super().__init__()

        self.main = nn.Sequential(
            nn.Conv2d(in_channels=depth, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        self.dueling = Dueling(
                nn.Sequential(linear_layer(3136, 512),
                              nn.ReLU(),
                              linear_layer(512, 1)),
                nn.Sequential(linear_layer(3136, 512),
                              nn.ReLU(),
                              linear_layer(512, actions))
            )

    def forward(self, x, advantages_only=False):
        f = self.main(x)
        return self.dueling(f, advantages_only=advantages_only)


class ImpalaCNNSmall(nn.Module):
    """
    Implementation of the small variant of the IMPALA CNN introduced in Espeholt et al. (2018).
    """
    def __init__(self, depth, actions, linear_layer):
        super().__init__()

        self.main = nn.Sequential(
            nn.Conv2d(in_channels=depth, out_channels=16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2),
            nn.ReLU(),
        )

        self.pool = torch.nn.AdaptiveMaxPool2d((6, 6))

        self.dueling = Dueling(
                nn.Sequential(linear_layer(1152, 256),
                              nn.ReLU(),
                              linear_layer(256, 1)),
                nn.Sequential(linear_layer(1152, 256),
                              nn.ReLU(),
                              linear_layer(256, actions))
            )

    def forward(self, x, advantages_only=False):
        f = self.main(x)
        f = self.pool(f)
        return self.dueling(f, advantages_only=advantages_only)


class ImpalaCNNResidual(nn.Module):
    """
    Simple residual block used in the large IMPALA CNN.
    """
    def __init__(self, depth, norm_func):
        super().__init__()

        self.relu = nn.ReLU()
        self.conv_0 = norm_func(nn.Conv2d(in_channels=depth, out_channels=depth, kernel_size=3, stride=1, padding=1))
        self.conv_1 = norm_func(nn.Conv2d(in_channels=depth, out_channels=depth, kernel_size=3, stride=1, padding=1))

    def forward(self, x):
        x_ = self.conv_0(self.relu(x))
        x_ = self.conv_1(self.relu(x_))
        return x+x_

class ImpalaCNNBlock(nn.Module):
    """
    Three of these blocks are used in the large IMPALA CNN.
    """
    def __init__(self, depth_in, depth_out, norm_func):
        super().__init__()

        self.conv = nn.Conv2d(in_channels=depth_in, out_channels=depth_out, kernel_size=3, stride=1, padding=1)
        self.max_pool = nn.MaxPool2d(3, 2, padding=1)
        self.residual_0 = ImpalaCNNResidual(depth_out, norm_func=norm_func)
        self.residual_1 = ImpalaCNNResidual(depth_out, norm_func=norm_func)

    def forward(self, x):
        x = self.conv(x)
        x = self.max_pool(x)
        x = self.residual_0(x)
        x = self.residual_1(x)
        return x


class ImpalaCNNLarge(nn.Module):
    """
    Implementation of the large variant of the IMPALA CNN introduced in Espeholt et al. (2018).
    """
    def __init__(self, in_depth, actions, linear_layer, model_size=1, spectral_norm=False):
        super().__init__()

        def identity(p): return p

        norm_func = torch.nn.utils.spectral_norm if (spectral_norm == 'all') else identity
        norm_func_last = torch.nn.utils.spectral_norm if (spectral_norm == 'last' or spectral_norm == 'all') else identity

        self.main = nn.Sequential(
            ImpalaCNNBlock(in_depth, 16*model_size, norm_func=norm_func),
            ImpalaCNNBlock(16*model_size, 32*model_size, norm_func=norm_func),
            ImpalaCNNBlock(32*model_size, 32*model_size, norm_func=norm_func_last),
            nn.ReLU()
        )

        self.pool = torch.nn.AdaptiveMaxPool2d((8, 8))

        self.dueling = Dueling(
            nn.Sequential(linear_layer(2048*model_size, 256),
                          nn.ReLU(),
                          linear_layer(256, 1)),
            nn.Sequential(linear_layer(2048*model_size, 256),
                          nn.ReLU(),
                          linear_layer(256, actions))
        )

    def forward(self, x, advantages_only=False):
        f = self.main(x)
        f = self.pool(f)
        return self.dueling(f, advantages_only=advantages_only)


class ConvNeXtAttoModel(nn.Module):
    """
    Implementation of the large variant of the IMPALA CNN introduced in Espeholt et al. (2018).
    """
    def __init__(self, in_depth, actions, linear_layer, spectral_norm=False, resolution=224, global_pool_type='max'):
        super().__init__()

        self.convnext_backbone = timm.create_model('convnext_atto', pretrained=False, in_chans=in_depth)
        self.convnext_backbone.head.global_pool = nn.Identity()
        self.convnext_backbone.head.norm = nn.Identity()
        self.convnext_backbone.head.flatten = nn.Identity()
        self.convnext_backbone.head.drop = nn.Identity()
        self.convnext_backbone.head.fc = nn.Identity()

        if global_pool_type == 'max':
            self.pool = torch.nn.AdaptiveMaxPool2d((3, 3))
        elif global_pool_type == 'avg':
            self.pool = torch.nn.AdaptiveAvgPool2d((3, 3))
        else:
            raise ValueError(f'Unknown value for global pool type {global_pool_type}')

        if spectral_norm == 'all':
            for stage in self.convnext_backbone.stages:
                for block in stage.blocks:
                    block.conv_dw = torch.nn.utils.spectral_norm(block.conv_dw)
                    block.mlp.fc1 = torch.nn.utils.spectral_norm(block.mlp.fc1)
                    block.mlp.fc2 = torch.nn.utils.spectral_norm(block.mlp.fc2)
        elif spectral_norm == 'all_mlp':
            for stage in self.convnext_backbone.stages:
                for block in stage.blocks:
                    block.mlp.fc1 = torch.nn.utils.spectral_norm(block.mlp.fc1)
                    block.mlp.fc2 = torch.nn.utils.spectral_norm(block.mlp.fc2)
        elif spectral_norm == 'all_dw':
            for stage in self.convnext_backbone.stages:
                for block in stage.blocks:
                    block.conv_dw = torch.nn.utils.spectral_norm(block.conv_dw)
        elif spectral_norm == 'last':
            stage = self.convnext_backbone.stages[-1]
            for block in stage.blocks:
                block.conv_dw = torch.nn.utils.spectral_norm(block.conv_dw)
                block.mlp.fc1 = torch.nn.utils.spectral_norm(block.mlp.fc1)
                block.mlp.fc2 = torch.nn.utils.spectral_norm(block.mlp.fc2)
        elif spectral_norm == 'last_mlp':
            stage = self.convnext_backbone.stages[-1]
            for block in stage.blocks:
                block.mlp.fc1 = torch.nn.utils.spectral_norm(block.mlp.fc1)
                block.mlp.fc2 = torch.nn.utils.spectral_norm(block.mlp.fc2)
        elif spectral_norm == 'last_dw':
            stage = self.convnext_backbone.stages[-1]
            for block in stage.blocks:
                block.conv_dw = torch.nn.utils.spectral_norm(block.conv_dw)

        self.dueling = Dueling(
            nn.Sequential(linear_layer(2880, 256),
                          nn.GELU(),
                          linear_layer(256, 1)),
            nn.Sequential(linear_layer(2880, 256),
                          nn.GELU(),
                          linear_layer(256, actions))
        )

    def forward(self, x, advantages_only=False):
        f = self.convnext_backbone(x)
        f = self.pool(f)
        return self.dueling(f, advantages_only=advantages_only)


class ImpalaNeXtCNNResidual(nn.Module):
    """
    Simple residual block used in the large IMPALA CNN.
    Changes from IMPALA: inc kernel size, gelu, layer norm, act pos
    """
    def __init__(self, depth, h, w, norm_func, layer_norm, activation_position):
        super().__init__()

        self.gelu1 = nn.GELU() if activation_position in ['before', 'both'] else nn.Identity()
        self.conv_0 = norm_func(nn.Conv2d(in_channels=depth, out_channels=depth, kernel_size=7, stride=1, padding=3))
        self.layer_nrom = nn.LayerNorm([depth, h, w]) if layer_norm else nn. Identity()
        self.conv_1 = norm_func(nn.Conv2d(in_channels=depth, out_channels=depth, kernel_size=7, stride=1, padding=3))
        self.gelu2 = nn.GELU() if activation_position in ['after', 'both'] else nn.Identity()

    def forward(self, x):
        x_ = self.gelu1(x)
        x_ = self.conv_0(x_)
        x_ = self.layer_nrom(x_)
        x_ = self.gelu2(x_)
        x_ = self.conv_1(x_)
        return x +  x_

class ImpalaNeXtCNNBlock(nn.Module):
    """
    Three of these blocks are used in the large IMPALA CNN.
    """
    def __init__(self, depth, h, w, norm_func, layer_norm, activation_pos):
        super().__init__()

        self.residual_0 = ImpalaNeXtCNNResidual(depth, h, w, norm_func, layer_norm, activation_pos)
        self.residual_1 = ImpalaNeXtCNNResidual(depth,h, w, norm_func, layer_norm, activation_pos)

    def forward(self, x):
        x = self.residual_0(x)
        x = self.residual_1(x)
        return x


class ImpalaNeXtDownsample(nn.Module):
    def __init__(self, depth_in, depth_out, w, h, convnext_like=False, layer_norm=False, blur_pool=False):
        super().__init__()
        if not convnext_like:
            self.layer_norm = nn.Identity()
            self.conv = nn.Conv2d(in_channels=depth_in, out_channels=depth_out, kernel_size=7, stride=1, padding=3)
            if not blur_pool:
                self.max_pool = nn.MaxPool2d(3, 2, padding=1)
            else:
                self.max_pool = nn.Sequential(
                    nn.MaxPool2d(3, 1, padding=1),
                    timm.models.layers.blur_pool.BlurPool2d(depth_out, 3, stride=2)
                )
        else:
            self.layer_norm = nn.LayerNorm([depth_in, h, w]) if layer_norm else nn.Identity()
            self.conv = nn.Conv2d(in_channels=depth_in, out_channels=depth_out, kernel_size=2, stride=2, padding=0)
            self.max_pool = nn.Identity()

    def forward(self, x):
        x = self.layer_norm(x)
        x = self.conv(x)
        x = self.max_pool(x)
        return x


class ImpalaNeXtPatchifyStem(nn.Module):
    def __init__(self, depth_in, depth_out, h, w, patch_size, layer_norm=False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=depth_in, out_channels=depth_out,
                              kernel_size=patch_size, stride=patch_size, padding=0)
        self.layer_norm = nn.LayerNorm([depth_out, h, w]) if layer_norm else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        return x


class ImpalaNeXtCNNLarge(nn.Module):
    """
    Implementation of the large variant of the IMPALA CNN introduced in Espeholt et al. (2018).
    """
    def __init__(self, in_depth, actions, linear_layer, model_size=1, spectral_norm=False, stem='orig', layer_norm=False, 
                    convnext_downsampling=False, activation_pos='after', blur_pool=False):
        super().__init__()

        def identity(p): return p

        norm_func = torch.nn.utils.spectral_norm if (spectral_norm == 'all') else identity
        norm_func_last = torch.nn.utils.spectral_norm if (spectral_norm == 'last' or spectral_norm == 'all') else identity

        if stem == 'orig':
            self.stem = ImpalaNeXtDownsample(in_depth, 16 * model_size, 84, 84, convnext_like=False, blur_pool=blur_pool)
        elif stem == 'patchify':
            self.stem = ImpalaNeXtPatchifyStem(in_depth, 16 * model_size, 84, 84, 2, layer_norm=layer_norm)
        else:
            raise ValueError(f'Unknown stem type: {stem}')

        self.main = nn.Sequential(
            ImpalaNeXtCNNBlock(16 * model_size, 42, 42, norm_func=norm_func, layer_norm=layer_norm, activation_pos=activation_pos),
            ImpalaNeXtDownsample(16 * model_size, 32*model_size, 42, 42, convnext_like=convnext_downsampling, layer_norm=layer_norm, blur_pool=blur_pool),
            ImpalaNeXtCNNBlock(32*model_size, 21, 21, norm_func=norm_func, layer_norm=layer_norm, activation_pos=activation_pos),
            ImpalaNeXtDownsample(32 * model_size, 32*model_size, 21, 21, convnext_like=convnext_downsampling, layer_norm=layer_norm, blur_pool=blur_pool),
            ImpalaNeXtCNNBlock(32 * model_size, 11, 11, norm_func=norm_func_last, layer_norm=layer_norm, activation_pos=activation_pos),
            nn.GELU()
        )

        self.pool = torch.nn.AdaptiveMaxPool2d((8, 8))

        self.dueling = Dueling(
            nn.Sequential(linear_layer(2048*model_size, 256),
                          nn.GELU(),
                          linear_layer(256, 1)),
            nn.Sequential(linear_layer(2048*model_size, 256),
                          nn.GELU(),
                          linear_layer(256, actions))
        )

    def forward(self, x, advantages_only=False):
        f = self.stem(x)
        f = self.main(f)
        f = self.pool(f)
        return self.dueling(f, advantages_only=advantages_only)


class ConvNeXtImpala(nn.Module):
    def __init__(self, in_depth, actions, linear_layer, width):
        super().__init__()
        self.model = timm.models.ConvNeXt(
                in_chans=in_depth,
                global_pool='avg',
                output_stride= 32,
                depths=(3, 3, 3, 0),
                dims=(16*width, 32*width, 64*width, 64*width),
                kernel_sizes=7,
                stem_type='patch',
                patch_size=3, # TODO calculate and make sure it is suitable
                conv_mlp=False,
                act_layer='gelu',
                norm_layer=None,
                drop_rate=0.0,
                drop_path_rate=0.0,
            )
        
        self.model.head.global_pool = nn.Identity()
        self.model.head.norm = nn.Identity()
        self.model.head.flatten = nn.Identity()
        self.model.head.fc = nn.Identity()
        self.model.head.dropout = nn.Identity()
        self.model.stages = self.model.stages[:-1]

        self.dueling = Dueling(
            nn.Sequential(linear_layer(7 * 7 * 64 * width, 512),
                          nn.GELU(),
                          linear_layer(512, 1)),
            nn.Sequential(linear_layer(7 * 7 * 64 * width, 512),
                          nn.GELU(),
                          linear_layer(512, actions))
        )

    def forward(self, x, advantages_only=False):
        f = self.model(x)
        return self.dueling(f, advantages_only=advantages_only)


def get_model(model_str, spectral_norm, resolution, global_pool_type):
    if model_str == 'nature': return NatureCNN
    elif model_str == 'dueling': return DuelingNatureCNN
    elif model_str == 'impala_small': return ImpalaCNNSmall
    elif model_str.startswith('impala_large:'):
        return partial(ImpalaCNNLarge, model_size=int(model_str[13:]), spectral_norm=spectral_norm)
    elif model_str.startswith('impalanext_large:'):
        return partial(ImpalaNeXtCNNLarge, model_size=int(model_str[17:]), spectral_norm=spectral_norm)
    elif model_str.startswith('impalanextv2_large:'):
        return partial(ImpalaNeXtCNNLarge, model_size=int(model_str[19:]), spectral_norm=spectral_norm, stem='patchify', convnext_downsampling=True, layer_norm=True)
    elif model_str.startswith('impalanextv3_large:'):
        return partial(ImpalaNeXtCNNLarge, model_size=int(model_str[19:]), spectral_norm=spectral_norm, stem='patchify', convnext_downsampling=True, layer_norm=False)
    elif model_str.startswith('impalanextv4_large:'):
        return partial(ImpalaNeXtCNNLarge, model_size=int(model_str[19:]), spectral_norm=spectral_norm, stem='orig', convnext_downsampling=True, layer_norm=False)
    elif model_str.startswith('impalanextv5_large:'):
        return partial(ImpalaNeXtCNNLarge, model_size=int(model_str[19:]), spectral_norm=spectral_norm, stem='orig', convnext_downsampling=False, layer_norm=False)
    elif model_str.startswith('impalanextv6_large:'):
        return partial(ImpalaNeXtCNNLarge, model_size=int(model_str[19:]), spectral_norm=spectral_norm, stem='orig',
                        convnext_downsampling=False, layer_norm=False, activation_pos='before')
    elif model_str.startswith('impalanextv7_large:'):
         return partial(ImpalaNeXtCNNLarge, model_size=int(model_str[19:]), spectral_norm=spectral_norm, stem='orig',
                        convnext_downsampling=True, layer_norm=False, activation_pos='before')
    elif model_str.startswith('impalanextv8_large:'):
         return partial(ImpalaNeXtCNNLarge, model_size=int(model_str[19:]), spectral_norm=spectral_norm, stem='patchify',
                        convnext_downsampling=True, layer_norm=False, activation_pos='before')
    elif model_str.startswith('impalanextv9_large:'):
         return partial(ImpalaNeXtCNNLarge, model_size=int(model_str[19:]), spectral_norm=spectral_norm, stem='patchify',
                        convnext_downsampling=False, layer_norm=False, activation_pos='before')
    elif model_str.startswith('impalanextv10_large:'):
         return partial(ImpalaNeXtCNNLarge, model_size=int(model_str[20:]), spectral_norm=spectral_norm, stem='patchify',
                        convnext_downsampling=False, layer_norm=True, activation_pos='before')
    elif model_str.startswith('impalanextv11_large:'):
         return partial(ImpalaNeXtCNNLarge, model_size=int(model_str[20:]), spectral_norm=spectral_norm, stem='patchify',
                        convnext_downsampling=False, layer_norm=False, activation_pos='before', blur_pool=True)
    elif model_str.startswith('convnext_atto'):
        return partial(ConvNeXtAttoModel, spectral_norm=spectral_norm, resolution=resolution)
    elif model_str.startswith('convnext_impala:'):
        return partial(ConvNeXtImpala, width=int(model_str[16:]))
