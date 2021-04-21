"""
通过给定的神经网络训练出相应大小的图像
"""
import math
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange


# helpers
def exists(val):
    return val is not None


# sin activation
class Sine(nn.Module):
    def __init__(self, w0=1.0):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)


# siren layer
class Siren(nn.Module):
    def __init__(
            self,
            dim_in,
            dim_out,
            w0=1.0,
            c=6.0,
            is_first=False,
            use_bias=True,
            activation=None,
    ):
        super().__init__()
        self.dim_in = dim_in
        self.is_first = is_first

        weight = torch.zeros(dim_out, dim_in)
        bias = torch.zeros(dim_out) if use_bias else None
        self.init_(weight, bias, c=c, w0=w0)

        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias) if use_bias else None
        self.activation = Sine(w0) if activation is None else activation

    def init_(self, weight, bias, c, w0):
        dim = self.dim_in

        w_std = (1 / dim) if self.is_first else (math.sqrt(c / dim) / w0)
        weight.uniform_(-w_std, w_std)

        if exists(bias):
            bias.uniform_(-w_std, w_std)

    def forward(self, x):
        #
        out = F.linear(x, self.weight, self.bias)
        # print(self.weight.shape)
        out = self.activation(out)
        return out


# siren network
class SirenNet(nn.Module):
    def __init__(
            self,
            dim_in,
            dim_hidden,
            dim_out,
            num_layers,
            w0=1.0,
            w0_initial=30.0,
            use_bias=True,
            final_activation=None,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dim_hidden = dim_hidden

        self.layers = nn.ModuleList([])
        for ind in range(num_layers):
            is_first = ind == 0  # 第一次为True其余为False
            layer_w0 = w0_initial if is_first else w0
            # 第一次（dim_hidden, dim_in),中间（dim_hidden, dim_hidden)
            layer_dim_in = dim_in if is_first else dim_hidden

            self.layers.append(
                Siren(
                    dim_in=layer_dim_in,
                    dim_out=dim_hidden,
                    w0=layer_w0,
                    use_bias=use_bias,
                    is_first=is_first,
                )
            )

        final_activation = (
            nn.Identity() if not exists(final_activation) else final_activation
        )
        self.last_layer = Siren(
            dim_in=dim_hidden,
            dim_out=dim_out,
            w0=w0,
            use_bias=use_bias,
            activation=final_activation,
        )

    def forward(self, x):
        # x (40000, 2)
        for layer in self.layers:
            x = layer(x)
        return self.last_layer(x)


'''
装饰器从给定的SirenNet中训练出特定高度和宽度的特定图像，然后再生成。
'''


class SirenWrapper(nn.Module):
    def __init__(self, net, image_width, image_height, latent_dim=None):
        super().__init__()
        # isinstance函数用来判断一个对象是否是一个已知的类型
        assert isinstance(net, SirenNet), "SirenWrapper must receive a Siren network"

        self.net = net
        self.image_width = image_width
        self.image_height = image_height

        # 线性生成数据
        tensors = [
            torch.linspace(-1, 1, steps=image_width),
            torch.linspace(-1, 1, steps=image_height),
        ]
        # meshgrid函数通过一个tensor生成2维的图像
        # stack （200， 200， 2）
        # (200, 200) 通过拼接得到（200， 200， 2）的mgrid
        mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
        # 重新排列（200， 200， 2） -> (40000, 2)
        mgrid = rearrange(mgrid, "h w c -> (h w) c")
        # 应该就是在内存中定一个常量，同时，模型保存和加载的时候可以写入和读出。
        self.register_buffer("grid", mgrid)

    def forward(self, img=None):
        # 从初始化中的图像中分离出一个，使其具有grad，而下一次的时候，coords不再前向传播
        coords = self.grid.clone().detach().requires_grad_()
        # 前向传播
        out = self.net(coords)

        out = rearrange(
            out, "(h w) c -> () c h w", h=self.image_height, w=self.image_width
        )

        return out
