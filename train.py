import os
import torch
from torch import optim
import cv2 as cv
from model.model import Model, Warping
from dataloader import Loader
from torch.utils.tensorboard import SummaryWriter
import configs
from metrics import calculate_accuracy


BATCH_SIZE = configs.BATCH_SIZE
VAL_BATCH_SIZE = configs.VAL_BATCH_SIZE
SEQ_LEN = configs.SEQ_LEN
Height, Width = [configs.H, configs.W]
height, width = [configs.h, configs.w]
SAVE_MODEL_PATH = configs.SAVEMODELPATH if not configs.FINETUNE else configs.SAVE_FINETUNE_MODELPATH


logger = SummaryWriter(log_dir=SAVE_MODEL_PATH+"/tensorboard")
def log_metrics(metrics_dict, prefix, epoch, iter):
    for key, value in metrics_dict.items():
        logger.add_scalar(f'{prefix}/{key}', value, iter)


def train(dataLoaderIns, valDataLoaderIns, model=None, finetune=False):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Running on device:", device)

    if model is None:
        model = Model(f=configs.f, 
                    m=configs.m, 
                    jitter=configs.jittering, 
                    depth_dilation=configs.depth_dilation).to(device)

    if configs.TRANSFER_LEARN:
        model.load_state_dict(
            torch.load(configs.TRANSFER_MODELPATH, map_location=device)["state_dict"]
        )

    optimizer = optim.Adam(model.parameters(), lr=configs.PRUNE_LEARNING_RATE if finetune else configs.LEARNING_RATE)
    lmbda = lambda e: configs.DECAY_RATE
    scheduler = optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lmbda)

    criterion = torch.nn.L1Loss(reduction='mean').to(device)
    smooth_criterion = torch.nn.L1Loss(reduction='mean').to(device)

    total_iter = 0

    for e in range(configs.EPOCHS):
        print("Training.....", end="\r")
        print("epochs :", e)

        loss_iter = 0
        avg_loss = 0
        avg_l1_loss = 0
        avg_distillation_loss = 0
        avg_smoothness_loss = 0

        for sample in dataLoaderIns.dataloader:

            optimizer.zero_grad()
            model.train()

            input_color = sample['color_lr'].to(device) # [B, seq_len, 3, h, w]
            input_depth = sample['depth_lr'].to(device) # [B, seq_len, 1, h, w]
            input_mv = sample['mv_lr'].to(device) # [B, seq_len, 2, h, w]
            input_jitter = sample['jitter_lr'].to(device) # [B, seq_len, 2, 1, 1]

            gt_color = sample['color_hr'].to(device) # [B, seq_len, 3, H, W]
            output_color = torch.zeros((BATCH_SIZE, SEQ_LEN, 3, Height, Width)).to(device)

            prev_color = sample['prev_color_hr'].to(device) # [B, 3, H, W]
            prev_jitter = sample['prev_jitter'].to(device) # [B, 2, 1, 1]
            prev_features = torch.zeros((BATCH_SIZE, 1, Height, Width)).to(device) # [B, 1, H, W]

            for seq in range(SEQ_LEN):
                color = input_color[:, seq]
                depth = input_depth[:, seq]
                jitter = input_jitter[:, seq]
                motion = input_mv[:, seq]

                prev_color, prev_features, _ = model(color, depth, jitter, prev_jitter, motion, prev_features, prev_color, device)
                output_color[:, seq] = prev_color

            loss_l1 = criterion(output_color, gt_color)
            loss = loss_l1

            avg_loss = (avg_loss * loss_iter + loss.detach().cpu().numpy()) / (loss_iter + 1)
            avg_l1_loss = (avg_l1_loss * loss_iter + loss_l1.detach().cpu().numpy()) / (loss_iter + 1)

            loss.backward()
            optimizer.step()
            
            if total_iter % configs.PRINT_LOSS == 0:
                print("iter", total_iter, "loss", avg_loss)
                log_metrics({"avg loss": avg_loss}, 'train', e+1, total_iter+1)
                avg_loss = 0
                avg_l1_loss = 0
                loss_iter = 0

            if total_iter % configs.CHECKPOINT == 0:
                torch.save(
                    {
                        "epoch": e + 1,
                        "iter": total_iter + 1,
                        "state_dict": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                    },
                    f"{SAVE_MODEL_PATH}/model.{e}.{total_iter}.pth.tar",
                )
                validate(model, valDataLoaderIns, criterion, device, e+1, total_iter+1)
            
            if total_iter in configs.DECAY_MILESTONE:
                scheduler.step()

            total_iter += 1
            loss_iter += 1

def validate(model, valDataLoaderIns, criterion, device, train_epochs, train_iters, model_warping=None, smooth_criterion=None):

    model.eval()
    iter = 0
    total_loss = 0
    total_PSNR = 0

    for sample in valDataLoaderIns.dataloader:

        input_color = sample['color_lr'].to(device) # [B, seq_len, 3, h, w]
        input_depth = sample['depth_lr'].to(device) # [B, seq_len, 1, h, w]
        input_mv = sample['mv_lr'].to(device) # [B, seq_len, 2, h, w]
        input_jitter = sample['jitter_lr'].to(device) # [B, seq_len, 2, 1, 1]

        gt_color = sample['color_hr'].to(device) # [B, seq_len, 3, H, W]
        output_color = torch.zeros((VAL_BATCH_SIZE, SEQ_LEN, 3, Height, Width)).to(device)

        prev_color = sample['prev_color_hr'].to(device) # [B, 3, H, W]
        prev_jitter = sample['prev_jitter'].to(device) # [B, 2, 1, 1]
        prev_features = torch.zeros((VAL_BATCH_SIZE, 1, Height, Width)).to(device) # [B, 1, H, W]

        for seq in range(SEQ_LEN):
            color = input_color[:, seq]
            depth = input_depth[:, seq]
            jitter = input_jitter[:, seq]
            motion = input_mv[:, seq]

            prev_color, prev_features, _ = model(color, depth, jitter, prev_jitter, motion, prev_features, prev_color, device)
            output_color[:, seq] = prev_color

        loss_l1 = criterion(output_color, gt_color)
        loss = loss_l1
        psnr = calculate_accuracy(output_color.detach().cpu().numpy() * 255, gt_color.detach().cpu().numpy() * 255)

        total_loss += loss.detach().cpu().numpy()
        total_PSNR += psnr
        
        iter += 1

    print("iter", train_iters, "eval loss:", total_loss/iter, "PSNR:", total_PSNR/iter)
    log_metrics({"avg loss": total_loss/iter, "avg PSNR": total_PSNR/iter}, 'val', train_epochs, train_iters)
    return


if __name__ == '__main__':
    if os.path.exists(SAVE_MODEL_PATH) is False:
    os.makedirs(SAVE_MODEL_PATH)

    train_data_loader = Loader(phase="train")
    val_data_loader = Loader(phase="val")
    train(train_data_loader, val_data_loader) 