import torch

model_name_list = ['RB_VIT_dir']

class DefaultConfig(object):
    class_name = 'bottle'
    data_root = r'data/mvtec_anomaly_detection'
    device = torch.device('cuda:0')
    model_name = model_name_list[0]
    batch_size = 4
    iter = 0
    niter = 400
    lr = 0.0001
    lr_decay = 0.90
    weight_decay = 1e-5
    momentum = 0.9
    nc = 3
    isTrain = True
    backbone_name = 'WideResnet50'
    referenc_img_file = f''
    resume = ''
    k = 4


if __name__ == '__main__':
    opt = DefaultConfig()
    opt.trai = 1
    print(opt.trai)