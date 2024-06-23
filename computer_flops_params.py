import torch
from thop import profile
from config import DefaultConfig
from models.TFA_Net_model import *
opt = DefaultConfig()
model = eval(opt.model_name)(opt)
model.eval()
input = torch.rand(1, 3, 256, 256)
flops, params = profile(model, (input, input, 'test'))
print('flops: ', str(flops/1e9)+'G', 'params: ', str(params/1e6)+'M')