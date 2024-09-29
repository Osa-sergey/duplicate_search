import torch
from .swinv2 import SwinTransformerV2

def load_swinv2(checkpoint_path):
    IMG_WIDTH = 256
    load_torchscript_model = torch.jit.load(checkpoint_path, map_location='cpu')
    checkpoint_state_dict = load_torchscript_model.state_dict()
    model = SwinTransformerV2(
        img_size=IMG_WIDTH,
        patch_size=4,
        window_size=16,
        num_heads=[4, 8, 16, 32],
        embed_dim=128,
        depths=[2, 2, 18, 2],
        pretrained_window_sizes=[12, 12, 12, 6],
        drop_path_rate=0.2,
        pretrained=None,
        output_dim=512,
        p=3.,
        use_checkpoint=False,
    )
    model.load_state_dict(checkpoint_state_dict)
    return model


def load_vit(checkpoint_path):
    from models.model_factory.backbones.sscd import SSCDModel
    vit_model = SSCDModel(
        name="vit_base_patch32_384",  # resnext101_32x4d  resnet50
        pool_param=3.,
        pool="gem",
        # pretrained=pretrained,
        use_classify=False,
        dims=(768, 512),
        add_head=True
    )
    vit_pretrain_state_dict = torch.jit.load(checkpoint_path, map_location='cpu').state_dict()
    vit_model.load_state_dict(vit_pretrain_state_dict)   
    return vit_model