import torch.nn as nn
from timm.models.vision_transformer import VisionTransformer, _cfg
from functools import partial
import torch

def tiny_deit(input_size, patch_size, n_channels, n_classes, dynamic_img_size=False):
	model = VisionTransformer(img_size=input_size, patch_size=patch_size, in_chans=n_channels, num_classes=n_classes, 
								embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True, 
								norm_layer=partial(nn.LayerNorm, eps=1e-6), dynamic_img_size=dynamic_img_size)
	model.default_cfg = _cfg()
	return model

class patchrot_classifier(nn.Module):
	def __init__(self, n_classes=4):
		super().__init__()

		self.fc = nn.Linear(192, n_classes)

		for m in self.modules():
			if isinstance(m, nn.Linear):
				nn.init.kaiming_normal_(m.weight)

	def forward(self, x):
		x = self.fc(x)
		return x
