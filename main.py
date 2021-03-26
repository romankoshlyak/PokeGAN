import argparse
import os
import json
import sys
import time

import torch
from torch import nn
from torch import optim
import matplotlib.pyplot as plt
import torchvision as tv
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np

from aegan import AEGAN
from aegan import ImageCache

BATCH_SIZE = 32
LATENT_DIM = 16
EPOCHS = 200000
RESULTS_DIR = "results"
CHECKPOINTS_DIR = os.path.join(RESULTS_DIR, "checkpoints")
CHECKPOINTS_PERIOD = 200
SAVE_IMAGES_PERIOD = 10

def save_img(image, filename):
    image = (image + 1) / 2.0
    just_save_img(image, filename)

def just_save_img(image, filename):
    if filename is not None:
        image = np.array(image*255, dtype=np.uint8)
        image = Image.fromarray(image)
        image.save(filename)

def gen_images(GAN, noise_fn, seed = 1, save_index = -1, filename = None):
    torch.manual_seed(seed)
    vec = noise_fn(100)
    images, confidence = GAN.generate_samples(vec)
    if filename is not None:
        filename = os.path.join(RESULTS_DIR, filename)
    if save_index == -1:
        ims = tv.utils.make_grid(images, normalize=True, nrow=10)
        fig = plt.figure(figsize=(16, 16), frameon=False)
        ax = fig.add_axes([0, 0, 1, 1])
        plt.axis('off')
        ax.imshow(ims.numpy().transpose((1,2,0)))
        plt.show()
    else:
        save_img(images[save_index], filename)

def load_gan():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    noise_fn = lambda x: torch.randn((x, LATENT_DIM), device=device)
    gan = AEGAN(
        LATENT_DIM,
        noise_fn,
        None,
        device=device,
        batch_size=BATCH_SIZE,
        checkpoints_dir=CHECKPOINTS_DIR
        )
    load_checkpoint(gan)
    return gan, noise_fn

def save_images(GAN, vec, filename):
    images, confidence = GAN.generate_samples(vec)
    ims = tv.utils.make_grid(images[:36], normalize=True, nrow=6,)
    ims = ims.numpy().transpose((1,2,0))
    ims = np.array(ims*255, dtype=np.uint8)
    image = Image.fromarray(ims)
    image.save(filename)

def find_last_checkpoint():
    last_file = None
    for filename in os.listdir(CHECKPOINTS_DIR):
        if filename.endswith(".pt") and (last_file is None or last_file < filename):
            last_file = filename
    if last_file is not None:
        last_epoch = int(last_file.split('.')[1])
        return os.path.join(CHECKPOINTS_DIR, last_file), last_epoch
    else:
        return None, -1

def load_checkpoint(model):
    last_file, last_epoch = find_last_checkpoint()
    print(f"Loading checkpoint {last_file}")
    if last_file is not None:
        model.load_state_dict(torch.load(last_file))
    return last_epoch

class Transform2Times(object):
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, sample):
        return [self.transform(sample), self.transform(sample)]

def main():
    os.makedirs("results/generated", exist_ok=True)
    os.makedirs("results/reconstructed", exist_ok=True)
    os.makedirs("results/checkpoints", exist_ok=True)

    root = os.path.join("data")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = tv.transforms.Compose([
            tv.transforms.RandomAffine(0, translate=(5/96, 5/96), fill=(255,255,255)),
            tv.transforms.ColorJitter(hue=0.5),
            tv.transforms.RandomHorizontalFlip(p=0.5),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize((0.5, 0.5, 0.5,), (0.5, 0.5, 0.5,))
            ])
    dataset = ImageFolder(
            root=root,
            transform=Transform2Times(transform)
            )
    dataloader = DataLoader(dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=2,
            drop_last=True
            )
    X = iter(dataloader)
    [test_ims, _], _ = next(X)
    while len(test_ims) < 36:
        [test_ims2, _], _ = next(X)
        test_ims = torch.cat((test_ims, test_ims2), 0)
    test_ims_show = tv.utils.make_grid(test_ims[:36], normalize=True, nrow=6,)
    test_ims_show = test_ims_show.numpy().transpose((1,2,0))
    test_ims_show = np.array(test_ims_show*255, dtype=np.uint8)
    image = Image.fromarray(test_ims_show)
    image.save("results/reconstructed/test_images.png")

    noise_fn = lambda x: torch.randn((x, LATENT_DIM), device=device)
    test_noise = noise_fn(36)
    gan = AEGAN(
        LATENT_DIM,
        noise_fn,
        dataloader,
        device=device,
        batch_size=BATCH_SIZE,
        checkpoints_dir=CHECKPOINTS_DIR
        )
    cache = ImageCache(BATCH_SIZE*4*1024)
    last_epoch = load_checkpoint(gan)
    start = time.time()
    for i in range(last_epoch+1, EPOCHS):
        elapsed = int(time.time() - start)
        elapsed = f"{elapsed // 3600:02d}:{(elapsed % 3600) // 60:02d}:{elapsed % 60:02d}"
        print(f"Epoch {i}; Elapsed time = {elapsed}s")
        gan.train_epoch(cache)
        if i > 0 and i % CHECKPOINTS_PERIOD == 0:
            torch.save(
                gan.state_dict(),
                os.path.join(CHECKPOINTS_DIR, f"gen.{i:05d}.pt"))

        if i % SAVE_IMAGES_PERIOD == 0:
            save_images(gan, test_noise,
                os.path.join("results", "generated", f"gen.{i:05d}.png"))

            with torch.no_grad():
                reconstructed = gan.generator(gan.encoder(test_ims.cuda())).cpu()
            reconstructed = tv.utils.make_grid(reconstructed[:36], normalize=True, nrow=6,)
            reconstructed = reconstructed.numpy().transpose((1,2,0))
            reconstructed = np.array(reconstructed*255, dtype=np.uint8)
            reconstructed = Image.fromarray(reconstructed)
            reconstructed.save(os.path.join("results", "reconstructed", f"gen.{i:05d}.png"))

    images = gan.generate_samples()
    ims = tv.utils.make_grid(images, normalize=True)
    plt.imshow(ims.numpy().transpose((1,2,0)))
    plt.show()

if __name__ == "__main__":
    main()
