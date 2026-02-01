from __future__ import annotations

from typing import Callable, List, Tuple

import torch
from torch import nn
from torchvision import models as tv_models

from models.exits import ConfidenceHead, ExitHead, MultiExitWrapper, TokenExitHead


def _infer_feature_shapes(feature_extractor: Callable, input_size: int = 224) -> List[torch.Tensor]:
    dummy = torch.zeros(1, 3, input_size, input_size)
    with torch.no_grad():
        feats = feature_extractor(dummy)
    return feats


def _build_exit_heads(features: List[torch.Tensor], num_classes: int, conf_head_type: str, token_use_cls: bool = True):
    exit_heads = []
    conf_heads = []
    for f in features:
        if f.dim() == 4:
            head = ExitHead(f.shape[1], num_classes)
            if conf_head_type == "mlp":
                conf_heads.append(ConfidenceHead(f.shape[1]))
        else:
            head = TokenExitHead(f.shape[-1], num_classes, use_cls=token_use_cls)
            if conf_head_type == "mlp":
                conf_heads.append(ConfidenceHead(f.shape[-1]))
        exit_heads.append(head)
    conf_heads = nn.ModuleList(conf_heads) if conf_heads else None
    return nn.ModuleList(exit_heads), conf_heads


def _resnet_extractor(model):
    def extractor(x):
        x = model.conv1(x)
        x = model.bn1(x)
        x = model.relu(x)
        x = model.maxpool(x)
        x = model.layer1(x)
        f2 = model.layer2(x)
        f3 = model.layer3(f2)
        f4 = model.layer4(f3)
        return [f2, f3, f4]

    return extractor


def _mobilenet_extractor(model):
    idxs = [len(model.features) // 3, 2 * len(model.features) // 3, len(model.features) - 1]

    def extractor(x):
        feats = []
        for i, block in enumerate(model.features):
            x = block(x)
            if i in idxs:
                feats.append(x)
        return feats

    return extractor


def _timm_features_model(name: str, pretrained: bool, out_indices: List[int]):
    import timm

    model = timm.create_model(name, pretrained=pretrained, features_only=True, out_indices=out_indices)
    return model


def _vit_extractor(model, exit_blocks: List[int]):
    def extractor(x):
        B = x.shape[0]
        x = model.patch_embed(x)
        if model.cls_token is not None:
            cls_tokens = model.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
        x = x + model.pos_embed
        x = model.pos_drop(x)
        feats = []
        for i, blk in enumerate(model.blocks):
            x = blk(x)
            if i in exit_blocks:
                feats.append(x)
        x = model.norm(x)
        # ensure last exit is normalized
        if exit_blocks and exit_blocks[-1] == len(model.blocks) - 1:
            feats[-1] = x
        return feats

    return extractor


def _swin_extractor(model, exit_layers: List[int]):
    def extractor(x):
        x = model.patch_embed(x)
        if hasattr(model, "absolute_pos_embed") and model.absolute_pos_embed is not None:
            x = x + model.absolute_pos_embed
        if hasattr(model, "pos_drop") and model.pos_drop is not None:
            x = model.pos_drop(x)
        feats = []
        for i, layer in enumerate(model.layers):
            x = layer(x)
            if i in exit_layers:
                feats.append(x)
        x = model.norm(x)
        if exit_layers and exit_layers[-1] == len(model.layers) - 1:
            feats[-1] = x
        return feats

    return extractor


def create_model(cfg_model, num_classes: int, conf_head_type: str = "maxprob") -> nn.Module:
    name = cfg_model.name
    pretrained = bool(cfg_model.pretrained)
    if name == "toy_mlp":
        class ToyMLP(nn.Module):
            def __init__(self, num_classes: int):
                super().__init__()
                self.fc1 = nn.Linear(2, 32)
                self.fc2 = nn.Linear(32, 32)
                self.fc3 = nn.Linear(32, 32)
                self.out1 = nn.Linear(32, num_classes)
                self.out2 = nn.Linear(32, num_classes)
                self.out3 = nn.Linear(32, num_classes)

            def forward(self, x):
                x1 = torch.relu(self.fc1(x))
                x2 = torch.relu(self.fc2(x1))
                x3 = torch.relu(self.fc3(x2))
                return [self.out1(x1), self.out2(x2), self.out3(x3)]

        return ToyMLP(num_classes)
    if name == "rn50":
        backbone = tv_models.resnet50(weights=tv_models.ResNet50_Weights.DEFAULT if pretrained else None)
        feature_extractor = _resnet_extractor(backbone)
    elif name == "mnv3s":
        backbone = tv_models.mobilenet_v3_small(weights=tv_models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None)
        feature_extractor = _mobilenet_extractor(backbone)
    elif name == "convnext_t":
        backbone = _timm_features_model("convnext_tiny", pretrained, out_indices=[1, 2, 3])
        feature_extractor = backbone
    elif name == "effb0":
        backbone = _timm_features_model("efficientnet_b0", pretrained, out_indices=[1, 2, 4])
        feature_extractor = backbone
    elif name == "vit_s":
        import timm

        backbone = timm.create_model("vit_small_patch16_224", pretrained=pretrained)
        num_blocks = len(backbone.blocks)
        exit_blocks = [num_blocks // 3 - 1, 2 * num_blocks // 3 - 1, num_blocks - 1]
        feature_extractor = _vit_extractor(backbone, exit_blocks)
    elif name == "swin_t":
        import timm

        backbone = timm.create_model("swin_tiny_patch4_window7_224", pretrained=pretrained)
        num_layers = len(backbone.layers)
        exit_layers = [max(0, num_layers - 3), max(0, num_layers - 2), num_layers - 1]
        feature_extractor = _swin_extractor(backbone, exit_layers)
    else:
        raise ValueError(f"Unknown model name: {name}")

    feats = _infer_feature_shapes(feature_extractor, input_size=224)
    token_use_cls = False if name == "swin_t" else True
    exit_heads, conf_heads = _build_exit_heads(feats, num_classes, conf_head_type, token_use_cls=token_use_cls)
    model = MultiExitWrapper(backbone, exit_heads, feature_extractor, conf_head_type=conf_head_type, conf_heads=conf_heads)
    return model
