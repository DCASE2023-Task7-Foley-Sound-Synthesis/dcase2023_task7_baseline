from typing import List
import argparse
import sys
import os

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from tqdm import tqdm

from vqvae import VQVAE
from scheduler import CycleScheduler

import audio2mel
from datasets import get_dataset_filelist


def train(epoch, loader, model, optimizer, scheduler, device):
    model.train()

    loader = tqdm(loader)

    criterion = nn.MSELoss()

    latent_loss_weight = 0.25

    mse_sum = 0
    mse_n = 0

    for i, (img, _, _, _) in enumerate(loader):
        model.zero_grad()

        img = img.to(device)

        out, latent_loss = model(img)
        recon_loss = criterion(out, img)
        latent_loss = latent_loss.mean()
        loss = recon_loss + latent_loss_weight * latent_loss
        loss.backward()

        if scheduler is not None:
            scheduler.step()
        optimizer.step()
        mse_sum = recon_loss.item() * img.shape[0]  # img.shape[0] = batch_size
        mse_n = img.shape[0]

        lr = optimizer.param_groups[0]["lr"]

        loader.set_description(
            (
                f"Epoch: {epoch + 1}; MSE: {recon_loss.item():.5f}; "
                f"latent: {latent_loss.item():.3f}; Avg MSE: {mse_sum / mse_n:.5f}; "
                f"lr: {lr:.5f}"
            )
        )

    latent_diff = latent_loss.item()
    return latent_diff, (mse_sum / mse_n)


def test(epoch, loader, model, optimizer, scheduler, device):
    model.eval()

    criterion = nn.MSELoss()

    mse_sum = 0
    mse_n = 0

    for i, (img, _, _, _) in enumerate(loader):
        model.zero_grad()

        img = img.to(device)

        out, latent_loss = model(img)
        recon_loss = criterion(out, img)
        latent_loss = latent_loss.mean()

        if scheduler is not None:
            scheduler.step()
        optimizer.step()
        part_mse_sum = recon_loss.item() * img.shape[0]  # img.shape[0] = batch_size
        part_mse_n = img.shape[0]
        comm = {"mse_sum": part_mse_sum, "mse_n": part_mse_n}

        for part in comm:
            mse_sum += part["mse_sum"]
            mse_n += part["mse_n"]

            # validation
            if i % 100 == 0:
                pass

    latent_diff = latent_loss.item()
    if (epoch + 1) % 10 == 0:
        print(
            f"\nTest_Epoch: {epoch + 1}; "
            f"latent: {latent_diff:.3f}; Avg MSE: {mse_sum / mse_n:.5f} \n"
        )
    return latent_diff, (mse_sum / mse_n)


def main(args):
    device = "cuda"

    train_file_list: List[dict] = get_dataset_filelist()

    train_set = audio2mel.Audio2Mel(
        train_file_list, 22050 * 4, 1024, 80, 256, 22050, 0, 8000
    )

    train_loader = DataLoader(
        train_set, batch_size=args.batch // args.n_gpu, num_workers=4, shuffle=True
    )

    print("training set size: " + str(len(train_set)))

    model = VQVAE().to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = None
    if args.sched == "cycle":
        scheduler = CycleScheduler(
            optimizer,
            args.lr,
            n_iter=len(train_loader) * args.epoch,
            momentum=None,
            warmup_proportion=0.05,
        )

    for i in range(args.epoch):
        train_latent_diff, train_average_loss = train(
            i, train_loader, model, optimizer, scheduler, device
        )

        # if dist.is_primary():

        torch.save(
            model.state_dict(), f"checkpoint/vqvae/vqvae_{str(i + 1).zfill(3)}.pt"
        )


if __name__ == "__main__":
    os.makedirs("checkpoint/vqvae/", exist_ok=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_gpu", type=int, default=1)

    port = (
        2**15
        + 2**14
        + hash(os.getuid() if sys.platform != "win32" else 1) % 2**14
    )

    parser.add_argument("--dist_url", default=f"tcp://127.0.0.1:{port}")

    parser.add_argument("--epoch", type=int, default=800)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--sched", type=str)

    args = parser.parse_args()

    print(args)

    main(args)
