import torch
import torch.nn as nn

class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, hp_lambda):
        ctx.hp_lambda = hp_lambda
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.hp_lambda, None

class GradientReversal(nn.Module):
    def __init__(self, hp_lambda):
        super(GradientReversal, self).__init__()
        self.hp_lambda = hp_lambda

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.hp_lambda)