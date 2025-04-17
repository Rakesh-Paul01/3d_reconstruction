import os
import argparse

import cv2
import torch
import torch.nn as nn

from config_utils import CONFIG
    
from dataloader import Pix3D_Recon_dataloader
from trainer import Recon_trainer
from model.loss import MonoSDFLoss
from model.network import Model

dirname = os.path.dirname(cv2.__file__)
plugin_path = os.path.join(dirname, 'plugins', 'platforms')
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = plugin_path


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('ssr training')
    parser.add_argument('--config', type=str, required=True, help='configure file for training or testing.')
    return parser.parse_args()

def run(cfg):
    torch.set_default_dtype(torch.float32)

    cfg.log_string('Data save path: %s' % (cfg.save_path))
    device = torch.device('cuda')

    train_loader = Pix3D_Recon_dataloader(cfg.config, 'train')
    test_loader = Pix3D_Recon_dataloader(cfg.config, 'val')

    loss = MonoSDFLoss(cfg.config)
    net = Model(cfg.config)

    params = [{"params": net.get_1x_lr_params(), "lr": float(cfg.config['optimizer']['lr'] / 10)},
                {"params": net.get_10x_lr_params(), "lr": float(cfg.config['optimizer']['lr'])}]
    optimizer = torch.optim.Adam(params, lr=float(cfg.config["optimizer"]["lr"]),
                                         betas=(cfg.config["optimizer"]["beta1"], cfg.config["optimizer"]["beta2"]))
    net = net.to(device)
    scheduler=torch.optim.lr_scheduler.MultiStepLR(optimizer,cfg.config["scheduler"]["milestone"],gamma=cfg.config['scheduler']['gamma'])

    trainer = Recon_trainer
    trainer(cfg, net, loss, optimizer,scheduler,train_loader=train_loader, test_loader=test_loader,device=device)

    cfg.log_string('Training finished.')

if __name__=="__main__":
    args=parse_args()
    cfg=CONFIG(args.config)
    cfg.update_config(args.__dict__)

    cfg.log_string('Loading configuration')
    cfg.log_string(cfg.config)
    cfg.write_config()

    cfg.log_string('Training begin.')
    run(cfg)
