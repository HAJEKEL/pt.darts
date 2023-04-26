""" Operations """
import torch
import torch.nn as nn
import genotypes as gt


OPS = {
    'none': lambda C, stride, affine: Zero(stride),
    'avg_pool_3x3': lambda C, stride, affine: PoolBN('avg', C, 3, stride, 1, affine=affine),
    'max_pool_3x3': lambda C, stride, affine: PoolBN('max', C, 3, stride, 1, affine=affine),
    'skip_connect': lambda C, stride, affine: \
        Identity() if stride == 1 else FactorizedReduce(C, C, affine=affine),
    'sep_conv_3x3': lambda C, stride, affine: SepConv(C, C, 3, stride, 1, affine=affine),
    'sep_conv_5x5': lambda C, stride, affine: SepConv(C, C, 5, stride, 2, affine=affine),
    'sep_conv_7x7': lambda C, stride, affine: SepConv(C, C, 7, stride, 3, affine=affine),
    'dil_conv_3x3': lambda C, stride, affine: DilConv(C, C, 3, stride, 2, 2, affine=affine), # 5x5
    'dil_conv_5x5': lambda C, stride, affine: DilConv(C, C, 5, stride, 4, 2, affine=affine), # 9x9
    'conv_7x1_1x7': lambda C, stride, affine: FacConv(C, C, 7, stride, 3, affine=affine),
    'mbconv_3_3': lambda C, stride, affine: MBConv(C, C, 3, stride, 1, 3, affine)
    # 'fused_mb_conv_3x3': lambda C, stride, affine: FusedMBConv(C, C, 3, stride, 1, affine=affine)
}

# OPS = {
#     'none': lambda C, stride, affine: Zero(stride),
#     'ds_conv_3x3': lambda C, stride, affine: DSConv(C, C, 3, stride, 1, affine=affine),
#     'mb_conv_3x3': lambda C, stride, affine: MBConv(C, C, 3, stride, 1, affine=affine),
#     'fused_mb_conv_3x3': lambda C, stride, affine: FusedMBConv(C, C, 3, stride, 1, affine=affine),
# }


def drop_path_(x, drop_prob, training):
    if training and drop_prob > 0.:
        keep_prob = 1. - drop_prob
        # per data point mask; assuming x in cuda.
        mask = torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob)
        x.div_(keep_prob).mul_(mask)

    return x

#efficientnet v1 https://github.com/kairos03/ProxylessNAS-Pytorch
def depthwise_conv(in_channels, kernel_size, stride, groups, affine):
    padding = kernel_size // 2
    return ConvBNReLU(in_channels, in_channels, kernel_size, stride, padding, groups, affine)

class ConvBNReLU(nn.Module):
  def __init__(self, C_in, C_out, kernel_size, stride, padding, groups=1, affine=True, activation=True):
    super(ConvBNReLU, self).__init__()

    self.conv = nn.Conv2d(C_in, C_out, kernel_size, stride, padding, groups=groups, bias=False)
    self.bn = nn.BatchNorm2d(C_out, affine=affine)
    if activation:
      self.act = nn.ReLU6()
    
  def forward(self, x):
    x = self.conv(x)
    x = self.bn(x)
    if hasattr(self, 'act'):
      x = self.act(x)
    return x
  
class MBConv(nn.Module):
  def __init__(self, C_in, C_out, kernel_size, stride, padding, expansion_factor, affine=True):
    super(MBConv, self).__init__()

    C_exp = C_in * expansion_factor
    self.res_connect = C_in == C_out and stride == 1

    self.op = nn.Sequential(
      ConvBNReLU(C_in, C_exp, 1, 1, 0, affine=affine),
      depthwise_conv(C_exp, kernel_size, stride, C_exp, affine=affine),
      ConvBNReLU(C_exp, C_out, 1, 1, 0, activation=False, affine=affine)
    )

  def forward(self, x):
    if self.res_connect:
      return self.op(x) + x
    else: 
      return self.op(x)

#efficientnet v2
class FusedMBConv(nn.Module):
  def __init__(self, C_in, C_out, kernel_size, stride, padding, expansion_factor, affine=True):
    super(FusedMBConv, self).__init__()

    C_exp = C_in * expansion_factor
    self.res_connect = C_in == C_out and stride == 1

    # Fused MBConv block
    self.op = nn.Sequential(
      ConvBNReLU(C_in, C_exp, kernel_size, stride, padding, groups=C_in, affine=affine),
      ConvBNReLU(C_exp, C_out, 1, 1, 0, activation=False, affine=affine)
    )

  def forward(self, x):
    if self.res_connect:
      return self.op(x) + x
    else: 
      return self.op(x)

    

class DropPath_(nn.Module):
    def __init__(self, p=0.):
        """ [!] DropPath is inplace module
        Args:
            p: probability of an path to be zeroed.
        """
        super().__init__()
        self.p = p

    def extra_repr(self):
        return 'p={}, inplace'.format(self.p)

    def forward(self, x):
        drop_path_(x, self.p, self.training)

        return x


class PoolBN(nn.Module):
    """
    AvgPool or MaxPool - BN
    """
    def __init__(self, pool_type, C, kernel_size, stride, padding, affine=True):
        """
        Args:
            pool_type: 'max' or 'avg'
        """
        super().__init__()
        if pool_type.lower() == 'max':
            self.pool = nn.MaxPool2d(kernel_size, stride, padding)
        elif pool_type.lower() == 'avg':
            self.pool = nn.AvgPool2d(kernel_size, stride, padding, count_include_pad=False)
        else:
            raise ValueError()

        self.bn = nn.BatchNorm2d(C, affine=affine)

    def forward(self, x):
        out = self.pool(x)
        out = self.bn(out)
        return out


class StdConv(nn.Module):
    """ Standard conv
    ReLU - Conv - BN
    """
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super().__init__()
        self.net = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(C_in, C_out, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(C_out, affine=affine)
        )

    def forward(self, x):
        return self.net(x)


class FacConv(nn.Module):
    """ Factorized conv
    ReLU - Conv(Kx1) - Conv(1xK) - BN
    """
    def __init__(self, C_in, C_out, kernel_length, stride, padding, affine=True):
        super().__init__()
        self.net = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(C_in, C_in, (kernel_length, 1), stride, padding, bias=False),
            nn.Conv2d(C_in, C_out, (1, kernel_length), stride, padding, bias=False),
            nn.BatchNorm2d(C_out, affine=affine)
        )

    def forward(self, x):
        return self.net(x)


class DilConv(nn.Module):
    """ (Dilated) depthwise separable conv
    ReLU - (Dilated) depthwise separable - Pointwise - BN

    If dilation == 2, 3x3 conv => 5x5 receptive field
                      5x5 conv => 9x9 receptive field
    """
    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
        super().__init__()
        self.net = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(C_in, C_in, kernel_size, stride, padding, dilation=dilation, groups=C_in,
                      bias=False),
            nn.Conv2d(C_in, C_out, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine)
        )

    def forward(self, x):
        return self.net(x)


class SepConv(nn.Module):
    """ Depthwise separable conv
    DilConv(dilation=1) * 2
    """
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super().__init__()
        self.net = nn.Sequential(
            DilConv(C_in, C_in, kernel_size, stride, padding, dilation=1, affine=affine),
            DilConv(C_in, C_out, kernel_size, 1, padding, dilation=1, affine=affine)
        )

    def forward(self, x):
        return self.net(x)


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class Zero(nn.Module):
    def __init__(self, stride):
        super().__init__()
        self.stride = stride

    def forward(self, x):
        if self.stride == 1:
            return x * 0.

        # re-sizing by stride
        return x[:, :, ::self.stride, ::self.stride] * 0.


class FactorizedReduce(nn.Module):
    """
    Reduce feature map size by factorized pointwise(stride=2).
    """
    def __init__(self, C_in, C_out, affine=True):
        super().__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.conv2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(C_out, affine=affine)

    def forward(self, x):
        x = self.relu(x)
        out = torch.cat([self.conv1(x), self.conv2(x[:, :, 1:, 1:])], dim=1)
        out = self.bn(out)
        return out


class MixedOp(nn.Module):
    """ Mixed operation """
    def __init__(self, C, stride):
        super().__init__()
        self._ops = nn.ModuleList()
        for primitive in gt.PRIMITIVES:
            op = OPS[primitive](C, stride, affine=False)
            self._ops.append(op)

    def forward(self, x, weights):
        """
        Args:
            x: input
            weights: weight for each operation
        """
        return sum(w * op(x) for w, op in zip(weights, self._ops))
