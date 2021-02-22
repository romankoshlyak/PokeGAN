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


BATCH_SIZE = 32
LATENT_DIM = 16
EPOCHS = 20000

def save_images(GAN, vec, filename):
    images = GAN.generate_samples(vec)
    ims = tv.utils.make_grid(images[:36], normalize=True, nrow=6,)
    ims = ims.numpy().transpose((1,2,0))
    ims = np.array(ims*255, dtype=np.uint8)
    image = Image.fromarray(ims)
    image.save(filename)

def find_last_checkpoint():
    last_file = None
    check_dir = "results/checkpoints"
    for filename in os.listdir(check_dir):
        if filename.endswith(".pt") and (last_file is None or last_file < filename):
            last_file = filename
    if last_file is not None:
        last_epoch = int(last_file.split('.')[1])
        return os.path.join(check_dir, last_file), last_epoch
    else:
        return None, -1

def load_checkpoint(model):
    last_file, last_epoch = find_last_checkpoint()
    print(f"Loading checkpoint {last_file}")
    if last_file is not None:
        model.load_state_dict(torch.load(last_file))
    return last_epoch

def main():
    os.makedirs("results/generated", exist_ok=True)
    os.makedirs("results/reconstructed", exist_ok=True)
    os.makedirs("results/checkpoints", exist_ok=True)

    root = os.path.join("data")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = tv.transforms.Compose([
            tv.transforms.RandomAffine(0, translate=(5/96, 5/96), fillcolor=(255,255,255)),
            tv.transforms.ColorJitter(hue=0.5),
            tv.transforms.RandomHorizontalFlip(p=0.5),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize((0.5, 0.5, 0.5,), (0.5, 0.5, 0.5,))
            ])
    dataset = ImageFolder(
            root=root,
            transform=transform
            )
    dataloader = DataLoader(dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=8,
            drop_last=True
            )
    X = iter(dataloader)
    test_ims1, _ = next(X)
    test_ims2, _ = next(X)
    test_ims = torch.cat((test_ims1, test_ims2), 0)
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
        )
    last_epoch = load_checkpoint(gan)
    start = time.time()
    for i in range(last_epoch+1, EPOCHS):
        while True:
            try:
                with open("pause.json") as f:
                    pause = json.load(f)
                if pause['pause'] == 0:
                    break
                print(f"Pausing for {pause['pause']} seconds")
                time.sleep(pause["pause"])
            except (KeyError, json.decoder.JSONDecodeError, FileNotFoundError):
                break
        elapsed = int(time.time() - start)
        elapsed = f"{elapsed // 3600:02d}:{(elapsed % 3600) // 60:02d}:{elapsed % 60:02d}"
        print(f"Epoch {i}; Elapsed time = {elapsed}s")
        gan.train_epoch(max_steps=100)
        if i > 0 and i % 50 == 0:
            torch.save(
                gan.state_dict(),
                os.path.join("results", "checkpoints", f"gen.{i:05d}.pt"))
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
