from models.model_MAE import *
from models.networks import *

from torch import nn


class RB_VIT_dir(nn.Module):
    def __init__(self, opt):
        super(RB_VIT_dir, self).__init__()

        if opt.backbone_name == 'D_VGG':
            self.Feature_extractor = D_VGG().eval()
            self.Roncon_model = RB_MAE_dir(in_chans=768)

        if opt.backbone_name == 'VGG':
            self.Feature_extractor = VGG().eval()
            self.Roncon_model = RB_MAE_dir(in_chans=960, patch_size=4)

        if opt.backbone_name == 'Resnet34':
            self.Feature_extractor = Resnet34().eval()
            self.Roncon_model = RB_MAE_dir(in_chans=512)

        if opt.backbone_name == 'Resnet50':
            self.Feature_extractor = Resnet50().eval()
            self.Roncon_model = RB_MAE_dir(in_chans=1856)

        if opt.backbone_name == 'WideResnet50':
            self.Feature_extractor = WideResNet50().eval()
            self.Roncon_model = RB_MAE_dir(in_chans=1856, patch_size=opt.k)

        if opt.backbone_name == 'Resnet101':
            self.Feature_extractor = Resnet101().eval()
            self.Roncon_model = RB_MAE_dir(in_chans=1856)

        if opt.backbone_name == 'WideResnet101':
            self.Feature_extractor = WideResnet101().eval()
            self.Roncon_model = RB_MAE_dir(in_chans=1856)

        if opt.backbone_name == 'MobileNet':
            self.Feature_extractor = MobileNet().eval()
            self.Roncon_model = RB_MAE_dir(in_chans=104)


    def forward(self, imgs, ref_imgs, stages):
        deep_feature = self.Feature_extractor(imgs)
        ref_deep_feature = self.Feature_extractor(ref_imgs)
        loss, pre_feature, _ = self.Roncon_model(deep_feature, ref_deep_feature)
        pre_feature_recon = self.Roncon_model.unpatchify(pre_feature)
        # vis_feature = [self.Roncon_model.unpatchify(i) for i in vis_feature]
        return deep_feature, ref_deep_feature, pre_feature_recon, loss

    def a_map(self, deep_feature, recon_feature):
        # recon_feature = self.Roncon_model.unpatchify(pre_feature)
        batch_size = recon_feature.shape[0]
        dis_map = torch.mean((deep_feature - recon_feature) ** 2, dim=1, keepdim=True)
        dis_map = nn.functional.interpolate(dis_map, size=(256, 256), mode="bilinear", align_corners=True).squeeze(1)
        dis_map = dis_map.clone().squeeze(0).cpu().detach().numpy()

        dir_map = 1 - torch.nn.CosineSimilarity()(deep_feature, recon_feature)
        dir_map = dir_map.reshape(batch_size, 1, 64, 64)
        dir_map = nn.functional.interpolate(dir_map, size=(256, 256), mode="bilinear", align_corners=True).squeeze(1)
        dir_map = dir_map.clone().squeeze(0).cpu().detach().numpy()
        return dis_map, dir_map

