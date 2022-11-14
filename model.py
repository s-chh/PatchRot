# import torch.nn as nn
# import torch.nn.functional as F
# import transformers
# import pdb
# import torch

# class encoder(nn.Module):
#     def __init__(self, args):
#         super(encoder, self).__init__()
#         self.args = args
#         vit_args = transformers.ViTConfig(hidden_size=256, num_hidden_layers=7, num_attention_heads=4, 
#                                             intermediate_size=512, hidden_act='gelu', hidden_dropout_prob=0.1, 
#                                             attention_probs_dropout_prob=0.0, initializer_range=0.02, 
#                                             layer_norm_eps=1e-12, is_encoder_decoder=False, image_size=self.args.pr_img_size,
#                                             patch_size=args.patch_size, num_channels=args.num_channels, qkv_bias=True)

#         model = transformers.ViTModel(vit_args)

#         self.cls_fc = nn.Sequential(*[model.pooler.dense, model.pooler.activation])
#         self.pr_fc = [nn.Sequential(*[model.pooler.dense, model.pooler.activation])]*self.args.n_pr_patches

#         model.pooler = nn.Identity()
#         self.encoder = model

#     def forward(self, x, prot=True):
#         if prot:
#             x = self.encoder(x)[0]

#             x_out = []
#             x_out.append(self.cls_fc(x[:, 0, :]))
#             for i in range(self.args.n_pr_patches):
#                 x_out.append(self.pr_fc[i](x[:, i + 1, :]))
#             x_out = torch.stack(x_out).permute(1, 0, 2)
#         else:
#             x = self.encoder(x, interpolate_pos_encoding=True)[0]
#             x_out = self.cls_fc(x[:, 0, :])
#         return x_out


# class classifier(nn.Module):
#     def __init__(self, args):
#         super(classifier, self).__init__()
#         self.args = args
#         self.fc = nn.Linear(256, self.args.num_classes)
        
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.kaiming_normal_(m.weight)

#     def forward(self, x):
#         x = self.fc(x)
#         return x


# class prot_classifier(nn.Module):
#     def __init__(self, args):
#         super(prot_classifier, self).__init__()
#         self.args = args
#         self.pr_fc = [nn.Linear(256, 4).cuda()] * (args.n_pr_patches + 1)
        
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.kaiming_normal_(m.weight)

#     def forward(self, x):
#         x_out = []
#         for i in range(self.args.n_pr_patches + 1):
#             x_out.append(self.pr_fc[i](x[:, i, :]))
#         x_out = torch.stack(x_out).permute(1, 0, 2)
#         return x_out



import torch.nn as nn
import torch.nn.functional as F
import transformers
import pdb
import torch

class get_out(torch.nn.Module):
    def forward(self, x):
        return x[0]


class encoder(nn.Module):
    def __init__(self, args):
        super(encoder, self).__init__()
        self.args = args
        vit_args = transformers.ViTConfig(hidden_size=256, num_hidden_layers=7, num_attention_heads=4, 
                                            intermediate_size=512, hidden_act='gelu', hidden_dropout_prob=0.1, 
                                            attention_probs_dropout_prob=0.0, initializer_range=0.02, 
                                            layer_norm_eps=1e-12, is_encoder_decoder=False, image_size=self.args.pr_img_size, 
                                            patch_size=args.patch_size, num_channels=args.num_channels, qkv_bias=True)

        model = transformers.ViTModel(vit_args)
        self.encoder = nn.Sequential(*[model.embeddings, model.encoder, get_out(), model.layernorm])
        self.fc = nn.Sequential(*[model.pooler.dense, model.pooler.activation])
        self.ss_fc = [nn.Sequential(*[model.pooler.dense, model.pooler.activation])]*self.args.n_pr_patches

    def forward(self, x, prot=True):
        if prot:
            x = self.encoder(x)

            x_out = []
            x_out.append(self.fc(x[:, 0, :]))
            for i in range(self.args.n_pr_patches):
                x_out.append(self.ss_fc[i](x[:, i+1, :]))
            x_out = torch.stack(x_out).permute(1, 0, 2)

        else:
            x = self.encoder[0](x, interpolate_pos_encoding=True)
            x = self.encoder[1](x)
            x = self.encoder[2](x)
            x = self.encoder[3](x)

            x = x[:, 0, :]
            x_out = self.fc(x)
            
        return x_out


class classifier(nn.Module):
    def __init__(self, args):
        super(classifier, self).__init__()
        self.args = args
        self.fc = nn.Linear(256, args.num_classes)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, x):
        x = self.fc(x)
        return x

class prot_classifier(nn.Module):
    def __init__(self, args):
        super(prot_classifier, self).__init__()
        self.args = args
        self.fc = nn.Linear(256, 4)
        self.ss_fc = [nn.Linear(256, 4).cuda()] * args.n_pr_patches
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, x):
        x_out = []
        x_out.append(self.fc(x[:, 0, :]))
        for i in range(self.args.n_pr_patches):
            x_out.append(self.ss_fc[i](x[:, i+1, :]))
        x_out = torch.stack(x_out).permute(1, 0, 2)
        return x_out
