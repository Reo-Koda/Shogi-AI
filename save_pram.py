from datetime import datetime
from torch import torch

def save_pram(model, optimizer, scheduler, epoch, val_loss, savePath="default"):
    if savePath == "default":
        dt_now = datetime.now()
        savePath = f"./pram/ckpt-{dt_now.strftime('%Y%m%d-%H%M%S')}-{model.__class__.__name__}-{int(val_loss * 1e6)}.pt"

    # モデルの保存
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
    }
    torch.save(ckpt, savePath)
    print(f"保存先のファイル : {savePath}", flush=True)