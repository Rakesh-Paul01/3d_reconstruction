import torch
from torch import nn
import numpy as np

from .embed import get_embedder
from utils_other import repeat_interleave


class ImplicitNetwork(nn.Module):
    def __init__(
        self,
        config,
        feature_vector_size,
        sdf_bounding_sphere,
        d_in,
        d_out,
        dims,
        geometric_init=True,
        bias=1.0,
        skip_in=(),
        weight_norm=True,
        multires=0,
        sphere_scale=1.0,
        inside_outside=False,
    ):
        super().__init__()

        self.use_global_encoder = True
        self.use_cls_encoder = True
        self.sdf_bounding_sphere = sdf_bounding_sphere
        self.sphere_scale = sphere_scale
        self.skip_in = skip_in

        self.embed_fn = None
        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, input_dims=d_in)
            self.embed_fn = embed_fn
            dims = [input_ch + 256] + dims + [d_out + feature_vector_size]
            if self.use_global_encoder:
                dims[0] += 256
            if self.use_cls_encoder:
                dims[0] += 9
            dims[skip_in[0]] = dims[0]  # adjust skip layer size
        else:
            dims = [d_in] + dims + [d_out + feature_vector_size]

        self.num_layers = len(dims)
        self.softplus = nn.Softplus(beta=100)

        for l in range(self.num_layers - 1):
            out_dim = dims[l + 1]
            layer = nn.Linear(dims[l], out_dim)

            if geometric_init:
                if l == self.num_layers - 2:
                    mean_val = np.sqrt(np.pi) / np.sqrt(dims[l])
                    init_mean = -mean_val if inside_outside else mean_val
                    torch.nn.init.normal_(layer.weight, mean=init_mean, std=0.0001)
                    torch.nn.init.constant_(layer.bias, bias if inside_outside else -bias)
                elif multires > 0 and l == 0:
                    torch.nn.init.constant_(layer.bias, 0.0)
                    torch.nn.init.constant_(layer.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(layer.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif multires > 0 and l in skip_in:
                    torch.nn.init.constant_(layer.bias, 0.0)
                    torch.nn.init.normal_(layer.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(layer.weight[:, -(dims[0] - 3):], 0.0)
                else:
                    torch.nn.init.constant_(layer.bias, 0.0)
                    torch.nn.init.normal_(layer.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                layer = nn.utils.weight_norm(layer)

            setattr(self, f"lin{l}", layer)

    def forward(self, input, latent_feature, cat_feature):
        if self.embed_fn is not None:
            input = self.embed_fn(input)

        x = input
        cat_feature = repeat_interleave(cat_feature, x.shape[0] // cat_feature.shape[0])
        x = torch.cat([x, cat_feature], dim=1)

        latent_feature = repeat_interleave(latent_feature, x.shape[0] // latent_feature.shape[0])
        skip_feature = None

        for l in range(self.num_layers - 1):
            lin = getattr(self, f"lin{l}")

            if l == 0:
                x = torch.cat([x, latent_feature], dim=1)
                skip_feature = x

            if l in self.skip_in:
                x = x + skip_feature

            x = lin(x)
            if l < self.num_layers - 2:
                x = self.softplus(x)

        return x

    def gradient(self, x, latent_feature, cat_feature):
        x.requires_grad_(True)
        y = self.forward(x, latent_feature, cat_feature)[:, :1]
        d_output = torch.ones_like(y)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        return gradients

    def get_outputs(self, x, latent_feature, cat_feature):
        x.requires_grad_(True)
        output = self.forward(x, latent_feature, cat_feature)
        sdf = output[:, :1]
        if self.sdf_bounding_sphere > 0.0:
            sphere_sdf = self.sphere_scale * (self.sdf_bounding_sphere - x.norm(2, 1, keepdim=True))
            sdf = torch.minimum(sdf, sphere_sdf)
        feature_vectors = output[:, 1:]
        d_output = torch.ones_like(sdf)
        gradients = torch.autograd.grad(
            outputs=sdf,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        return sdf, feature_vectors, gradients

    def get_sdf_vals(self, x, latent_feature, cat_feature):
        sdf = self.forward(x, latent_feature, cat_feature)[:, :1]
        if self.sdf_bounding_sphere > 0.0:
            sphere_sdf = self.sphere_scale * (self.sdf_bounding_sphere - x.norm(2, 1, keepdim=True))
            sdf = torch.minimum(sdf, sphere_sdf)
        return sdf
class RenderingNetwork(nn.Module):
    def __init__(
            self,
            feature_vector_size,
            mode,
            d_in,
            d_out,
            dims,
            weight_norm=True,
            multires_view=0,
            per_image_code=False
    ):
        super().__init__()

        self.mode = mode
        dims = [d_in + feature_vector_size] + dims + [d_out]

        self.embedview_fn = None
        if multires_view > 0:
            embedview_fn, input_ch = get_embedder(multires_view)
            self.embedview_fn = embedview_fn
            dims[0] += (input_ch - 3)

        print("rendering network architecture:")
        print(dims)

        self.num_layers = len(dims)

        for l in range(0, self.num_layers - 1):
            out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, f"lin{l}", lin)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, points, normals, view_dirs, feature_vectors, indices):
        if self.embedview_fn is not None:
            view_dirs = self.embedview_fn(view_dirs)

        x = torch.cat([view_dirs, feature_vectors], dim=-1)

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, f"lin{l}")
            x = lin(x)
            if l < self.num_layers - 2:
                x = self.relu(x)

        return self.sigmoid(x)
