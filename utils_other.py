import os
import random

import numpy as np
import urllib
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torch.nn import init
import torch.nn.functional as F
import functools
from skimage import measure
import torchvision
import trimesh
from PIL import Image

from utils import *



# replace for torch.repeat_interleave
def repeat_interleave(input, repeats, dim=0):
    """
    Repeat interleave along axis 0
    torch.repeat_interleave is currently very slow
    https://github.com/pytorch/pytorch/issues/31980
    """
    output = input.unsqueeze(1).expand(-1, repeats, *input.shape[1:])
    return output.reshape(-1, *input.shape[1:])


def read_obj_point(obj_path):
    with open(obj_path, 'r') as f:
        content_list = f.readlines()
        point_list = [line.rstrip("\n").lstrip("v ").split(" ") for line in content_list]
        for point in point_list:
            for i in range(3):
                point[i] = float(point[i])
        return np.array(point_list)


def write_obj_point(points,obj_path):
    with open(obj_path,'w') as f:
        for i in range(points.shape[0]):
            point=points[i]
            write_line="v %.4f %.4f %.4f\n"%(point[0],point[1],point[2])
            f.write(write_line)
    return


def load_checkpoint(fpath, model):
    ckpt = torch.load(fpath, map_location='cpu')

    if 'net' in ckpt:
        ckpt = ckpt['net']
    load_dict = {}
    for k, v in ckpt.items():
        if k.startswith('module.'):
            k_ = k.replace('module.', '')
            load_dict[k_] = v
        else:
            load_dict[k] = v

    modified = {}  # backward compatibility to older naming of architecture blocks
    for k, v in load_dict.items():
        if k.startswith('adaptive_bins_layer.embedding_conv.'):
            k_ = k.replace('adaptive_bins_layer.embedding_conv.',
                           'adaptive_bins_layer.conv3x3.')
            modified[k_] = v

        elif k.startswith('adaptive_bins_layer.patch_transformer.embedding_encoder'):

            k_ = k.replace('adaptive_bins_layer.patch_transformer.embedding_encoder',
                           'adaptive_bins_layer.patch_transformer.embedding_convPxP')
            modified[k_] = v
            # del load_dict[k]
        else:
            modified[k] = v  # else keep the original

    if 'encoder.latent' in modified:
        del modified['encoder.latent']

    model.load_state_dict(modified)
    return model


def worker_init_fn(worker_id):
    random_data = os.urandom(4)
    base_seed = int.from_bytes(random_data, byteorder="big")
    np.random.seed(base_seed + worker_id)


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """

    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find(
                'BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


def conv3x3(in_planes, out_planes, strd=1, padding=1, bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3,
                     stride=strd, padding=padding, bias=bias)


def imageSpaceRotation(xy, rot):
    '''
    args:
        xy: (B, 2, N) input
        rot: (B, 2) x,y axis rotation angles

    rotation center will be always image center (other rotation center can be represented by additional z translation)
    '''
    disp = rot.unsqueeze(2).sin().expand_as(xy)
    return (disp * xy).sum(dim=1)


def cal_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0):
    """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028

    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( | |gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss

    Returns the gradient penalty loss
    """
    if lambda_gp > 0.0:
        if type == 'real':  # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            alpha = torch.rand(real_data.shape[0], 1)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(
                *real_data.shape)
            alpha = alpha.to(device)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp  # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None


def get_norm_layer(norm_type='instance', group_norm_groups=32):
    """Return a normalization layer
    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none
    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'group':
        norm_layer = functools.partial(nn.GroupNorm, group_norm_groups)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class ConvBlock(nn.Module):
    def __init__(self, in_planes, out_planes, norm='batch'):
        super(ConvBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, int(out_planes / 2))
        self.conv2 = conv3x3(int(out_planes / 2), int(out_planes / 4))
        self.conv3 = conv3x3(int(out_planes / 4), int(out_planes / 4))

        if norm == 'batch':
            self.bn1 = nn.BatchNorm2d(in_planes)
            self.bn2 = nn.BatchNorm2d(int(out_planes / 2))
            self.bn3 = nn.BatchNorm2d(int(out_planes / 4))
            self.bn4 = nn.BatchNorm2d(in_planes)
        elif norm == 'group':
            self.bn1 = nn.GroupNorm(32, in_planes)
            self.bn2 = nn.GroupNorm(32, int(out_planes / 2))
            self.bn3 = nn.GroupNorm(32, int(out_planes / 4))
            self.bn4 = nn.GroupNorm(32, in_planes)

        if in_planes != out_planes:
            self.downsample = nn.Sequential(
                self.bn4,
                nn.ReLU(True),
                nn.Conv2d(in_planes, out_planes,
                          kernel_size=1, stride=1, bias=False),
            )
        else:
            self.downsample = None

    def forward(self, x):
        residual = x

        out1 = self.bn1(x)
        out1 = F.relu(out1, True)
        out1 = self.conv1(out1)

        out2 = self.bn2(out1)
        out2 = F.relu(out2, True)
        out2 = self.conv2(out2)

        out3 = self.bn3(out2)
        out3 = F.relu(out3, True)
        out3 = self.conv3(out3)

        out3 = torch.cat((out1, out2, out3), 1)

        if self.downsample is not None:
            residual = self.downsample(residual)

        out3 += residual

        return out3

# refer: https://blog.csdn.net/ytusdc/article/details/125529881
def fix_random_seed(seed=1029):
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


# from utils import rend_util


avg_pool_3d = torch.nn.AvgPool3d(2, stride=2)
upsample = torch.nn.Upsample(scale_factor=2, mode='nearest')


@torch.no_grad()
def get_surface_sliding(path, epoch, model, img, intrinsics, extrinsics, model_input, ground_truth, resolution=512, grid_boundary=[-2.0, 2.0], return_mesh=False, delta=0, level=0, eval_gt=False, export_color_mesh=False):
    assert resolution % 256 == 0

    model.encoder(img)                           # img: (B, C, H, W)
    image_shape = torch.empty(2).cuda()          # (W, H)
    image_shape[0] = img.shape[-1]               # W
    image_shape[1] = img.shape[-2]               # H

    batch_size = img.shape[0]

    resN = resolution
    cropN = 256
    level = 0.0
    N = resN // cropN

    # grid_min = grid_boundary.min(dim=1)[0]          # .min -> (values, indices)   .min[0] -> values
    # grid_max = grid_boundary.max(dim=1)[0]
    grid_min = np.array([-1, -1, -1])
    grid_max = np.array([1, 1, 1])
    xs = np.linspace(grid_min[0]-delta, grid_max[0]+delta, N+1)
    ys = np.linspace(grid_min[1]-delta, grid_max[1]+delta, N+1)
    zs = np.linspace(grid_min[2]-delta, grid_max[2]+delta, N+1)

    # for evaluation, align InstPIFu size
    bbox_scale_value = 2.0 / (2.0 - ground_truth['voxel_padding'][0])

    meshes = []
    for i in range(N):
        for j in range(N):
            for k in range(N):

                x_min, x_max = xs[i], xs[i+1]
                y_min, y_max = ys[j], ys[j+1]
                z_min, z_max = zs[k], zs[k+1]

                x = np.linspace(x_min, x_max, cropN)
                y = np.linspace(y_min, y_max, cropN)
                z = np.linspace(z_min, z_max, cropN)

                xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
                points = torch.tensor(np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T, dtype=torch.float).cuda().to(torch.float32)          # in cube coords

                def evaluate(points):
                    z = []
                    for _, pnts in enumerate(torch.split(points, 100000, dim=0)):
                        # get model object coords
                        model_obj = pnts / model_input['none_equal_scale'] + model_input['centroid']
                        scene_obj = model_obj * model_input['scene_scale']                                                          # [N, 3]
                        scene_obj = scene_obj[None, None, ...]                                                                      # [1, 1, N, 3]
                        world_coords = obj2world(scene_obj, model_input['obj_rot'], model_input['obj_tran'])                        # [1, 1, N, 3]

                        latent_feature, cat_feature = get_latent_feature(model, world_coords.reshape(-1, 3), intrinsics, extrinsics, model_input)

                        sdf = model.implicit_network(pnts, latent_feature, cat_feature)[:, 0]
                        z.append(sdf)
                    z = torch.cat(z, axis=0)
                    return z

                def evaluate_gt(points, ground_truth):
                    model_obj = points / model_input['none_equal_scale'] + model_input['centroid']
                    scene_obj = model_obj * model_input['scene_scale']                                                          # [N, 3]
                    scene_obj = scene_obj[None, None, ...]                                                                      # [1, 1, N, 3]
                    world_coords = obj2world(scene_obj, model_input['obj_rot'], model_input['obj_tran'])                        # [1, 1, N, 3]

                    sdf_gt = get_sdf_gt_worldcoords(world_coords, ground_truth)

                    z = sdf_gt.squeeze(1)
                    return z

                # construct point pyramids
                points = points.reshape(cropN, cropN, cropN, 3).permute(3, 0, 1, 2)
                points_pyramid = [points]
                for _ in range(3):            
                    points = avg_pool_3d(points[None])[0]
                    points_pyramid.append(points)
                points_pyramid = points_pyramid[::-1]

                # evalute pyramid with mask
                mask = None
                threshold = 2 * (x_max - x_min)/cropN * 8
                for pid, pts in enumerate(points_pyramid):
                    coarse_N = pts.shape[-1]
                    pts = pts.reshape(3, -1).permute(1, 0).contiguous()

                    if mask is None:
                        if eval_gt:
                            pts_sdf = evaluate_gt(pts, ground_truth)
                        else:
                            pts_sdf = evaluate(pts)
                    else:
                        mask = mask.reshape(-1)
                        pts_to_eval = pts[mask]
                        #import pdb; pdb.set_trace()
                        if pts_to_eval.shape[0] > 0:
                            if eval_gt:
                                pts_sdf_eval = evaluate_gt(pts_to_eval.contiguous(), ground_truth)
                            else:
                                pts_sdf_eval = evaluate(pts_to_eval.contiguous())
                            pts_sdf[mask] = pts_sdf_eval
                        # print("ratio", pts_to_eval.shape[0] / pts.shape[0])

                    if pid < 3:
                        # update mask
                        mask = torch.abs(pts_sdf) < threshold
                        mask = mask.reshape(coarse_N, coarse_N, coarse_N)[None, None]
                        mask = upsample(mask.float()).bool()

                        pts_sdf = pts_sdf.reshape(coarse_N, coarse_N, coarse_N)[None, None]
                        pts_sdf = upsample(pts_sdf)
                        pts_sdf = pts_sdf.reshape(-1)               # [N, ]

                    threshold /= 2.


                z = pts_sdf.detach().cpu().numpy()                  # [N,]

                if (not (np.min(z) > level or np.max(z) < level)):
                    z = z.astype(np.float32)

                    verts, faces, normals, values = measure.marching_cubes(
                        volume=z.reshape(cropN, cropN, cropN), #.transpose([1, 0, 2]),
                        level=level,
                        spacing=(
                                (x_max - x_min)/(cropN-1),
                                (y_max - y_min)/(cropN-1),
                                (z_max - z_min)/(cropN-1)))

                    # print(np.array([x_min, y_min, z_min]))
                    # print(verts.min(), verts.max())
                    verts = verts + np.array([x_min, y_min, z_min])     # in cube coords

                    if not export_color_mesh:
                        # for evaluation
                        verts = verts * bbox_scale_value.detach().cpu().numpy()

                    meshcrop = trimesh.Trimesh(verts, faces, normals)

                    #meshcrop.export(f"{i}_{j}_{k}.ply")
                    meshes.append(meshcrop)

    combined = trimesh.util.concatenate(meshes)

    if return_mesh:
        return combined
    else:
        combined.export('{0}/surface_{1}.ply'.format(path, epoch), 'ply')    


def plot_normal_maps(normal_maps, ground_true, path, epoch, img_res, indices, ray_mask):

    normal_maps = (normal_maps[0].view(img_res[0], img_res[1], -1)).cpu().detach().numpy()
    ray_mask_map = (ray_mask[0].view(img_res[0], img_res[1], -1)).cpu().detach().numpy()

    normal_maps = (normal_maps * 255).astype(np.uint8)
    normal_maps_temp = Image.fromarray(normal_maps)
    normal_maps = normal_maps_temp.convert('RGBA')

    # ray_mask_map = Image.fromarray(ray_mask_map[:, :, 0].astype(np.uint8) * 255)
    ray_mask_map = Image.fromarray((ray_mask_map[:, :, 0] * 255).astype(np.uint8))
    ray_mask_map = ray_mask_map.convert('L')
    normal_maps.putalpha(ray_mask_map)

    ground_true = ground_true.cuda()            # [B, N, 3]

    normal_maps_plot = lin2img(ground_true, img_res)

    tensor = torchvision.utils.make_grid(normal_maps_plot,
                                         scale_each=False,
                                         normalize=False).cpu().detach().numpy()
    tensor = tensor.transpose(1, 2, 0)
    scale_factor = 255
    tensor = (tensor * scale_factor).astype(np.uint8)

    img = Image.fromarray(tensor)
    return img, normal_maps


def plot_images(rgb_points, ground_true, path, epoch, img_res, indices, exposure=False, ray_mask=None):

    rgb_map = (rgb_points[0].view(img_res[0], img_res[1], -1)).cpu().detach().numpy()
    ray_mask_map = (ray_mask[0].view(img_res[0], img_res[1], -1)).cpu().detach().numpy()

    rgb_map = (rgb_map * 255).astype(np.uint8)
    rgb_map_temp = Image.fromarray(rgb_map)
    rgb_map = rgb_map_temp.convert('RGBA')

    # ray_mask_map = Image.fromarray(ray_mask_map[:, :, 0].astype(np.uint8) * 255)
    ray_mask_map = Image.fromarray((ray_mask_map[:, :, 0] * 255).astype(np.uint8))
    ray_mask_map = ray_mask_map.convert('L')
    rgb_map.putalpha(ray_mask_map)


    ground_true = ground_true.cuda()
    ground_true = lin2img(ground_true, img_res)

    tensor = torchvision.utils.make_grid(ground_true,
                                         scale_each=False,
                                         normalize=False,).cpu().detach().numpy()

    tensor = tensor.transpose(1, 2, 0)
    scale_factor = 255
    tensor = (tensor * scale_factor).astype(np.uint8)

    img = Image.fromarray(tensor)
    return img, rgb_map, rgb_map_temp


def plot_depth_maps(depth_maps, ground_true, path, epoch, img_res, indices, ray_mask):

    depth_maps = (depth_maps[0].view(img_res[0], img_res[1])).cpu().detach().numpy()
    ray_mask_map = (ray_mask[0].view(img_res[0], img_res[1], -1)).cpu().detach().numpy()

    # depth normalize
    max_depth = np.max(depth_maps)
    depth_maps = depth_maps / max_depth

    depth_maps = (depth_maps * 150).astype(np.uint8)
    depth_maps_temp = Image.fromarray(depth_maps)
    depth_maps = depth_maps_temp.convert('RGB')

    ray_mask_map = Image.fromarray((ray_mask_map[:, :, 0] * 255).astype(np.uint8))
    ray_mask_map = ray_mask_map.convert('L')
    depth_maps.putalpha(ray_mask_map)

    ground_true = ground_true.numpy()

    # depth normalize
    max_depth = np.max(ground_true)
    ground_true = ground_true / max_depth

    ground_true = ground_true[0].reshape(img_res[0], img_res[1])
    ground_true = (ground_true * 150).astype(np.uint8)
    ground_true = Image.fromarray(ground_true)
    ground_true = ground_true.convert('RGB')

    return ground_true, depth_maps


def lin2img(tensor, img_res):
    batch_size, num_samples, channels = tensor.shape
    return tensor.permute(0, 2, 1).view(batch_size, channels, img_res[0], img_res[1])


def split_input(model_input, total_pixels, n_pixels=10000):
    '''
     Split the input to fit Cuda memory for large resolution.
     Can decrease the value of n_pixels in case of cuda out of memory error.
     '''
    split = []
    for i, indx in enumerate(torch.split(torch.arange(total_pixels).cuda(), n_pixels, dim=0)):
        data = model_input.copy()
        data['uv'] = torch.index_select(model_input['uv'], 1, indx)
        if 'object_mask' in data:
            data['object_mask'] = torch.index_select(model_input['object_mask'], 1, indx)
        if 'depth' in data:
            data['depth'] = torch.index_select(model_input['depth'], 1, indx)
        split.append(data)
    return split


def merge_output(res, total_pixels, batch_size):
    ''' Merge the split output. '''
    model_outputs = {}
    for entry in res[0]:
        if res[0][entry] is None:
            continue
        if len(res[0][entry].shape) == 1:
            model_outputs[entry] = torch.cat([r[entry].reshape(batch_size, -1, 1) for r in res],
                                             1).reshape(batch_size * total_pixels)
        else:
            model_outputs[entry] = torch.cat([r[entry].reshape(batch_size, -1, r[entry].shape[-1]) for r in res],
                                             1).reshape(batch_size * total_pixels, -1)

    return model_outputs
