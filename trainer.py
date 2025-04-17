import os
import time
import datetime

import torch
from torch.utils.tensorboard import SummaryWriter

from utils import get_psnr

def Recon_trainer(cfg, model, loss, optimizer, scheduler, train_loader, test_loader, device):
    start_t = time.time()
    config = cfg.config

    log_dir = cfg.save_path
    os.makedirs(log_dir, exist_ok=True)

    cfg.write_config()
    tb_logger = SummaryWriter(log_dir)

    start_epoch = 0
    iter = 0
    scheduler.last_epoch = start_epoch

    model.train()
    min_eval_loss = float('inf')

    for e in range(start_epoch, config['other']['nepoch']):
        torch.cuda.empty_cache()
        cfg.log_string("Switch Phase to Train")
        model.train()
        for batch_id, (indices, model_input, ground_truth) in enumerate(train_loader):
            for key in model_input:
                model_input[key] = model_input[key].cuda().float()

            optimizer.zero_grad()
            model_outputs = model(model_input, indices)
            loss_output = loss(model_outputs, ground_truth, e)

            total_loss = loss_output['total_loss']
            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.7)
            total_norm = torch.sqrt(sum((p.grad.data.norm(2) ** 2 for p in model.parameters()
                                         if p.grad is not None and p.requires_grad)))
            optimizer.step()

            psnr = get_psnr(model_outputs['rgb_values'], ground_truth['rgb'].cuda().reshape(-1, 3))
            msg = (
                f"{str(datetime.timedelta(seconds=round(time.time() - start_t)))},[epoch {e}] "
                f"({batch_id + 1}/{len(train_loader)}): total_loss = {total_loss.item()}, "
                + ", ".join([f"{k} = {v.item()}" for k, v in loss_output.items() if k != 'total_loss']) +
                f", psnr = {psnr.item()}, beta = {model.module.density.get_beta().item()}, "
                f"alpha = {1. / model.module.density.get_beta().item()}"
            )
            cfg.log_string(msg)

            # Log to tensorboard
            for k, v in loss_output.items():
                tb_logger.add_scalar(f'Loss/{k}', v.item(), iter)
            tb_logger.add_scalar('Loss/grad_norm', total_norm, iter)
            tb_logger.add_scalar('Statistics/beta', model.module.density.get_beta().item(), iter)
            tb_logger.add_scalar('Statistics/alpha', 1. / model.module.density.get_beta().item(), iter)
            tb_logger.add_scalar('Statistics/psnr', psnr.item(), iter)
            tb_logger.add_scalar("train/lr", optimizer.param_groups[0]['lr'], iter)

            iter += 1

        # Evaluation
        if e % config['other']['model_save_interval'] == 0:
            model.eval()
            eval_loss, eval_loss_info = 0, {}
            cfg.log_string("Switch Phase to Test")

            with torch.no_grad():
                for batch_id, (indices, model_input, ground_truth) in enumerate(test_loader):
                    torch.cuda.empty_cache()
                    for key in model_input:
                        model_input[key] = model_input[key].cuda().float()

                    model_outputs = model(model_input, indices)
                    loss_output = loss(model_outputs, ground_truth, e)
                    total_loss = loss_output['total_loss']
                    eval_loss += total_loss.item()

                    for k, v in loss_output.items():
                        if "total" not in k:
                            eval_loss_info.setdefault(k, 0)
                            eval_loss_info[k] += v.mean().item()

                    psnr = get_psnr(model_outputs['rgb_values'], ground_truth['rgb'].cuda().reshape(-1, 3))
                    msg = (
                        f"Validation {str(datetime.timedelta(seconds=round(time.time() - start_t)))},[epoch {e}] "
                        f"({batch_id + 1}/{len(test_loader)}): total_loss = {total_loss.item()}, "
                        + ", ".join([f"{k} = {v.item()}" for k, v in loss_output.items() if k != 'total_loss']) +
                        f", psnr = {psnr.item()}, beta = {model.module.density.get_beta().item()}, "
                        f"alpha = {1. / model.module.density.get_beta().item()}"
                    )
                    cfg.log_string(msg)

            avg_eval_loss = eval_loss / (batch_id + 1)
            for k in eval_loss_info:
                eval_loss_info[k] /= (batch_id + 1)
                tb_logger.add_scalar("eval/" + k, eval_loss_info[k], e)
            cfg.log_string(f'avg_eval_loss is {avg_eval_loss}')
            tb_logger.add_scalar('eval/eval_loss', avg_eval_loss, e)

            # Scheduler step
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(avg_eval_loss)
            else:
                scheduler.step()
