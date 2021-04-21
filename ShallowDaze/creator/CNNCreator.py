"""
通过巻积神经网络生成图像
"""

import math
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange


# helpers

def exists(val):
    return val is not None


def cast_tuple(val, repeat=1):
    # 如果是tuple类型就返回
    return val if isinstance(val, tuple) else ((val,) * repeat)


class CnnNet(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self):
        # TODO： 构建具有快速拟合性质的CNN神经网络
        pass


'''
装饰器从给定的CnnNet网络中训练出特定高度和宽度的特定图像，然后再生成。
'''


class CnnWrapper(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self):
        # TODO： 调用CnnNet训练出图像
        pass
