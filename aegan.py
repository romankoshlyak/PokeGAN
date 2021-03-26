import os
import json
import sys
import time
import copy

import torch
from torch import nn
from torch import optim
import torchvision as tv
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np

ALPHA_COPY_IMAGE = 1.0
ALPHA_RECONSTRUCT_IMAGE = 0.0
ALPHA_RECONSTRUCT_LATENT = 1.0
ALPHA_DISCRIMINATE_IMAGE = 1.0
ALPHA_DISCRIMINATE_LATENT = 1.0

class ImageCache(object):
    def __init__(self, cache_size):
        self.cache_size = cache_size
        self.used_size = 0
        self.cache = None

    def add_images(self, images):
        images = images.cpu()
        if self.cache is None:
            self.cache = images[0:1].expand(self.cache_size, -1, -1, -1).clone()
        add_size = min(images.size(0), self.cache_size-self.used_size)
        insert_size = images.size(0)-add_size
        if add_size > 0:
            self.cache[self.used_size:self.used_size+add_size] = images
            self.used_size += add_size
        if insert_size > 0:
            self.cache[torch.randperm(self.cache_size)[:insert_size]] = images[images.size(0)-insert_size:].clone()

    def get_images(self, size):
        return self.cache[torch.randperm(self.used_size)[:size]]

class Generator(nn.Module):
    """A generator for mapping a latent space to a sample space.

    Input shape: (?, latent_dim)
    Output shape: (?, 3, 96, 96)
    """

    def __init__(self, latent_dim: int = 8):
        """Initialize generator.

        Args:
            latent_dim (int): latent dimension ("noise vector")
        """
        super().__init__()
        self.latent_dim = latent_dim
        self._init_modules()

    def build_colourspace(self, input_dim: int, output_dim: int):
        """Build a small module for selecting colours."""
        colourspace = nn.Sequential(
            nn.Linear(
                input_dim,
                128,
                bias=True),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),

            nn.Linear(
                128,
                64,
                bias=True),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),

            nn.Linear(
                64,
                output_dim,
                bias=True),
            nn.Tanh(),
            )
        return colourspace

    def _init_modules(self):
        """Initialize the modules."""
        projection_widths = [8, 8, 8, 8, 8, 8, 8]
        self.projection_dim = sum(projection_widths) + self.latent_dim
        self.projection = nn.ModuleList()
        for index, i in enumerate(projection_widths):
            self.projection.append(
                nn.Sequential(
                    nn.Linear(
                        self.latent_dim + sum(projection_widths[:index]),
                        i,
                        bias=True,
                        ),
                    nn.BatchNorm1d(8),
                    nn.LeakyReLU(),
                    )
                )
        self.projection_upscaler = nn.Upsample(scale_factor=3)

        self.colourspace_r = self.build_colourspace(self.projection_dim, 16)
        self.colourspace_g = self.build_colourspace(self.projection_dim, 16)
        self.colourspace_b = self.build_colourspace(self.projection_dim, 16)
        self.colourspace_upscaler = nn.Upsample(scale_factor=96)

        self.seed = nn.Sequential(
            nn.Linear(
                self.projection_dim,
                512*3*3,
                bias=True),
            nn.BatchNorm1d(512*3*3),
            nn.LeakyReLU(),
            )

        self.upscaling = nn.ModuleList()
        self.conv = nn.ModuleList()

        self.upscaling.append(nn.Upsample(scale_factor=2))
        self.conv.append(nn.Sequential(
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(
                in_channels=(512)//4,
                out_channels=512,
                kernel_size=3,
                stride=1,
                padding=0,
                bias=True
                ),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            ))

        self.upscaling.append(nn.Upsample(scale_factor=2))
        self.conv.append(nn.Sequential(
            nn.ZeroPad2d((1, 2, 1, 2)),
            nn.Conv2d(
                in_channels=(512 + self.projection_dim)//4,
                out_channels=256,
                kernel_size=4,
                stride=1,
                padding=0,
                bias=True
                ),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            ))

        self.upscaling.append(nn.Upsample(scale_factor=2))
        self.conv.append(nn.Sequential(
            nn.ZeroPad2d((1, 2, 1, 2)),
            nn.Conv2d(
                in_channels=(256 + self.projection_dim)//4,
                out_channels=256,
                kernel_size=4,
                stride=1,
                padding=0,
                bias=True
                ),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            ))

        self.upscaling.append(nn.Upsample(scale_factor=2))
        self.conv.append(nn.Sequential(
            nn.ZeroPad2d((1, 2, 1, 2)),
            nn.Conv2d(
                in_channels=(256 + self.projection_dim)//4,
                out_channels=256,
                kernel_size=4,
                stride=1,
                padding=0,
                bias=True
                ),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            )),

        self.upscaling.append(nn.Upsample(scale_factor=2))
        self.conv.append(nn.Sequential(
            nn.ZeroPad2d((1, 2, 1, 2)),
            nn.Conv2d(
                in_channels=(256 + self.projection_dim)//4,
                out_channels=64,
                kernel_size=4,
                stride=1,
                padding=0,
                bias=True
                ),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            ))

        self.upscaling.append(nn.Upsample(scale_factor=1))
        self.conv.append(nn.Sequential(
            nn.ZeroPad2d((2, 2, 2, 2)),
            nn.Conv2d(
                in_channels=64,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=0,
                bias=True
                ),
            nn.Softmax(dim=1),
            ))

    def forward(self, input_tensor):
        """Forward pass; map latent vectors to samples."""
        last = input_tensor
        for module in self.projection:
            projection = module(last)
            last = torch.cat((last, projection), -1)
        projection = last

        intermediate = self.seed(projection)
        intermediate = intermediate.view((-1, 512, 3, 3))

        projection_2d = projection.view((-1, self.projection_dim, 1, 1))
        projection_2d = self.projection_upscaler(projection_2d)

        for i, (conv, upscaling) in enumerate(zip(self.conv, self.upscaling)):
            if i + 1 != len(self.upscaling):
                if i > 0:
                    intermediate = torch.cat((intermediate, projection_2d), 1)
                intermediate = torch.nn.functional.pixel_shuffle(intermediate, 2)
            intermediate = conv(intermediate)
            projection_2d = upscaling(projection_2d)

        r_space = self.colourspace_r(projection)
        r_space = r_space.view((-1, 16, 1, 1))
        r_space = self.colourspace_upscaler(r_space)
        r_space = intermediate * r_space
        r_space = torch.sum(r_space, dim=1, keepdim=True)

        g_space = self.colourspace_g(projection)
        g_space = g_space.view((-1, 16, 1, 1))
        g_space = self.colourspace_upscaler(g_space)
        g_space = intermediate * g_space
        g_space = torch.sum(g_space, dim=1, keepdim=True)

        b_space = self.colourspace_b(projection)
        b_space = b_space.view((-1, 16, 1, 1))
        b_space = self.colourspace_upscaler(b_space)
        b_space = intermediate * b_space
        b_space = torch.sum(b_space, dim=1, keepdim=True)

        output = torch.cat((r_space, g_space, b_space), dim=1)

        return output


class Encoder(nn.Module):
    """An Encoder for encoding images as latent vectors.

    Input shape: (?, 3, 96, 96)
    Output shape: (?, latent_dim)
    """

    def __init__(self, device: str = "cpu", latent_dim: int = 8):
        """Initialize encoder.

        Args:
            device: chich GPU or CPU to use.
            latent_dim: output dimension
        """
        super().__init__()
        self.device = device
        self.latent_dim = latent_dim
        self._init_modules()

    def _init_modules(self):
        """Initialize the modules."""
        down_channels = [3, 64, 128, 256, 512]
        self.down = nn.ModuleList()
        for i in range(len(down_channels)-1):
            self.down.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=down_channels[i],
                        out_channels=down_channels[i+1],
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        bias=True,
                        ),
                    nn.BatchNorm2d(down_channels[i+1]),
                    nn.LeakyReLU(),
                    )
                )

        self.reducer = nn.Sequential(
            nn.Conv2d(
                in_channels=down_channels[-1],
                out_channels=down_channels[-2],
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
                ),
            nn.BatchNorm2d(down_channels[-2]),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2)
            )

        up_channels = [256, 128, 64, 64, 64]
        scale_factors = [2, 2, 2, 1]
        self.up = nn.ModuleList()
        for i in range(len(up_channels)-1):
            self.up.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=up_channels[i] + down_channels[-2-i],
                        out_channels=up_channels[i+1],
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=True,
                        ),
                    nn.BatchNorm2d(up_channels[i+1]),
                    nn.LeakyReLU(),
                    nn.Upsample(scale_factor=scale_factors[i]),
                    )
                )

        down_again_channels = [64+3, 64, 64, 64, 64]
        self.down_again = nn.ModuleList()
        for i in range(len(down_again_channels)-1):
            self.down_again.append(
                nn.Conv2d(
                    in_channels=down_again_channels[i],
                    out_channels=down_again_channels[i+1],
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=True,
                    )
                )
            self.down_again.append(nn.BatchNorm2d(down_again_channels[i+1]))
            self.down_again.append(nn.LeakyReLU())

        self.projection = nn.Sequential(
            nn.Linear(
                512*6*6 + 64*6*6,
                256,
                bias=True,
                ),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),

            nn.Linear(
                256,
                128,
                bias=True,
                ),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),

            nn.Linear(
                128,
                self.latent_dim,
                bias=True,
                ),
            )

    def forward(self, input_tensor):
        """Forward pass; map latent vectors to samples."""
        rv = torch.randn(input_tensor.size(), device=self.device) * 0.02
        augmented_input = input_tensor + rv
        intermediate = augmented_input
        intermediates = [augmented_input]
        for module in self.down:
            intermediate = module(intermediate)
            intermediates.append(intermediate)
        intermediates = intermediates[:-1][::-1]

        down = intermediate.view(-1, 6*6*512)

        intermediate = self.reducer(intermediate)

        for index, module in enumerate(self.up):
            intermediate = torch.cat((intermediate, intermediates[index]), 1)
            intermediate = module(intermediate)

        intermediate = torch.cat((intermediate, input_tensor), 1)

        for module in self.down_again:
            intermediate = module(intermediate)

        intermediate = intermediate.view(-1, 6*6*64)
        intermediate = torch.cat((down, intermediate), -1)

        projected = self.projection(intermediate)

        return projected


class DiscriminatorImage(nn.Module):
    """A discriminator for discerning real from generated images.

    Input shape: (?, 3, 96, 96)
    Output shape: (?, 1)
    """

    def __init__(self, input_channels = 3, device="cpu"):
        """Initialize the discriminator."""
        super().__init__()
        self.device = device
        self._init_modules(input_channels)

    def _init_modules(self, input_channels):
        """Initialize the modules."""
        mult = 1
        down_channels = [input_channels, 64*mult, 128*mult, 256*mult, 512*mult]
        self.down = nn.ModuleList()
        leaky_relu = nn.LeakyReLU()
        self.drop = nn.Dropout2d(p=0.2)
        for i in range(4):
            self.down.append(
                nn.Conv2d(
                    in_channels=down_channels[i],
                    out_channels=down_channels[i+1],
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=True,
                    )
                )
            self.down.append(nn.BatchNorm2d(down_channels[i+1]))
            self.down.append(leaky_relu)

        self.classifier = nn.ModuleList()
        self.width = down_channels[-1] * 6**2
        self.classifier.append(nn.Linear(self.width, 1))
        self.classifier.append(nn.Sigmoid())

    def forward(self, input_tensor):
        """Forward pass; map latent vectors to samples."""
        #rv = torch.randn(input_tensor.size(), device=self.device) * 0.02
        intermediate = input_tensor# + rv
        for module in self.down:
            intermediate = module(intermediate)
            intermediate = self.drop(intermediate)
            #rv = torch.randn(intermediate.size(), device=self.device) * 0.02 + 1
            #intermediate *= rv

        intermediate = intermediate.view(-1, self.width)

        for module in self.classifier:
            intermediate = module(intermediate)

        return intermediate


class DiscriminatorLatent(nn.Module):
    """A discriminator for discerning real from generated vectors.

    Input shape: (?, latent_dim)
    Output shape: (?, 1)
    """

    def __init__(self, latent_dim=8, device="cpu"):
        """Initialize the Discriminator."""
        super().__init__()
        self.latent_dim = latent_dim
        self.device = device
        self._init_modules()

    def _init_modules(self, depth=7, width=8):
        """Initialize the modules."""
        self.pyramid = nn.ModuleList()
        for i in range(depth):
            self.pyramid.append(
                nn.Sequential(
                    nn.Linear(
                        self.latent_dim + width*i,
                        width,
                        bias=True,
                        ),
                    nn.BatchNorm1d(width),
                    nn.LeakyReLU(),
                    )
                )

        self.classifier = nn.ModuleList()
        self.classifier.append(nn.Linear(depth*width + self.latent_dim, 1))
        self.classifier.append(nn.Sigmoid())

    def forward(self, input_tensor):
        """Forward pass; map latent vectors to samples."""
        last = input_tensor
        for module in self.pyramid:
            projection = module(last)
            rv = torch.randn(projection.size(), device=self.device) * 0.02 + 1
            projection *= rv
            last = torch.cat((last, projection), -1)
        for module in self.classifier:
            last = module(last)
        return last

class DetailsInfo(object):
    def __init__(self):
        self.data = {}
    def log(self, key, value, mult = 1):
        key_parts = key.split('_')
        for part_ind in range(len(key_parts)):
            sub_key = '_'.join(key_parts[:part_ind+1])
            for _ in range(mult):
                self.data.setdefault(part_ind, {}).setdefault(sub_key, []).append(value.item())

    def printStats(self, level):
        data = self.data.setdefault(level, {})
        for key in sorted(data.keys()):
            print(key, np.array(data[key]).mean())

class AEGAN(nn.Module):
    """An Autoencoder Generative Adversarial Network for making pokemon."""

    def __init__(self, latent_dim, noise_fn, dataloader,
                 batch_size=32, device='cpu', checkpoints_dir = None):
        """Initialize the AEGAN.
        Args:
            latent_dim: latent-space dimension. Must be divisible by 4.
            noise_fn: function f(num: int) -> pytorch tensor, (latent vectors)
            dataloader: a pytorch dataloader for loading images
            batch_size: training batch size. Must match that of dataloader
            device: cpu or CUDA
        """
        super().__init__()
        assert latent_dim % 4 == 0
        self.latent_dim = latent_dim
        self.device = device
        self.noise_fn = noise_fn
        self.dataloader = dataloader
        self.batch_size = batch_size
        self.checkpoints_dir = checkpoints_dir

        self.criterion_gen = nn.BCELoss()
        self.criterion_recon_image = nn.L1Loss()
        self.criterion_recon_latent = nn.MSELoss()
        self.target_ones = torch.ones((batch_size, 1), device=device)
        self.target_zeros = torch.zeros((batch_size, 1), device=device)
        self._init_generator()
        self._init_encoder()
        self._init_dx()
        self._init_dz()

    def _init_generator(self):
        self.generator = Generator(latent_dim=self.latent_dim)
        self.generator = self.generator.to(self.device)
        self.optim_g = optim.Adam(self.generator.parameters(),
                                  lr=2e-4, betas=(0.5, 0.999),
                                  weight_decay=1e-8)

    def _init_encoder(self):
        self.encoder = Encoder(latent_dim=self.latent_dim, device=self.device)
        self.encoder = self.encoder.to(self.device)
        self.optim_e = optim.Adam(self.encoder.parameters(),
                                  lr=2e-4, betas=(0.5, 0.999),
                                  weight_decay=1e-8)

    def _init_dx(self):
        self.discriminator_image = DiscriminatorImage(device=self.device).to(self.device)
        self.optim_di = optim.Adam(self.discriminator_image.parameters(),
                                   lr=1e-4, betas=(0.5, 0.999),
                                   weight_decay=1e-8)
        self.discriminator_image2 = DiscriminatorImage(input_channels = 6, device=self.device).to(self.device)
        self.optim_di2 = optim.Adam(self.discriminator_image2.parameters(),
                                   lr=1e-4, betas=(0.5, 0.999),
                                   weight_decay=1e-8)

    def _init_dz(self):
        self.discriminator_latent = DiscriminatorLatent(
            latent_dim=self.latent_dim,
            device=self.device,
            ).to(self.device)
        self.optim_dl = optim.SGD(self.discriminator_latent.parameters(),
                                   lr=1e-3)


    def generate_samples(self, latent_vec=None, num=None):
        """Sample images from the generator.

        Images are returned as a 4D tensor of values between -1 and 1.
        Dimensions are (number, channels, height, width). Returns the tensor
        on cpu.

        Args:
            latent_vec: A pytorch latent vector or None
            num: The number of samples to generate if latent_vec is None

        If latent_vec and num are None then use self.batch_size
        random latent vectors.
        """
        num = self.batch_size if num is None else num
        latent_vec = self.noise_fn(num) if latent_vec is None else latent_vec
        with torch.no_grad():
            samples = self.generator(latent_vec)
            confidence = self.discriminator_image(samples)
        samples = samples.cpu()  # move images to cpu
        return samples, confidence

    def get_model_files(self):
        model_files = []
        for filename in os.listdir(self.checkpoints_dir):
            if filename.endswith(".pt"):
                model_files.append(filename)

        return [os.path.join(self.checkpoints_dir, model_file) for model_file in sorted(model_files)]
    
    def calc_discriminators_stats(self, X, X_transformed, model, details_info):
        with torch.no_grad():
            Z = self.noise_fn(self.batch_size)
            X_hat = self.generator(Z)
            Z_hat = self.encoder(X)
            X_tilde = self.generator(Z_hat)
            Z_tilde = self.encoder(X_hat)

            X_copy = torch.cat((X, X_transformed), dim=1)
            X_copy_rolled = torch.cat((X, torch.roll(X_transformed, 1, 0)), dim=1)
            X_copy_tilde = torch.cat((X, X_tilde), dim=1)
            X_copy_tilde_rolled = torch.cat((X, torch.roll(X_tilde, 1, 0)), dim=1)

            X_copy_confidence = model.discriminator_image2(X_copy)
            X_not_copy_confidence = model.discriminator_image2(X_copy_rolled)
            X_copy_tilde_confidence = model.discriminator_image2(X_copy_tilde)
            X_not_copy_tilde_confidence = model.discriminator_image2(X_copy_tilde_rolled)

            X_copy_loss = self.criterion_gen(X_copy_confidence, self.target_ones)
            X_not_copy_loss = self.criterion_gen(X_not_copy_confidence, self.target_zeros)
            X_copy_tilde_loss = self.criterion_gen(X_copy_tilde_confidence, self.target_zeros)
            X_not_copy_tilde_loss = self.criterion_gen(X_not_copy_tilde_confidence, self.target_zeros)
            details_info.log('discImageCopyLoss_Copy', X_copy_loss)
            details_info.log('discImageCopyLoss_NotCopy', X_not_copy_loss)
            details_info.log('discImageCopyLoss_TildeCopy', X_copy_tilde_loss)
            details_info.log('discImageCopyLoss_TildeNotCopy', X_not_copy_tilde_loss)

            X_confidence = model.discriminator_image(X)
            X_hat_confidence = model.discriminator_image(X_hat)
            X_tilde_confidence = model.discriminator_image(X_tilde)

            discImageLoss_X = self.criterion_gen(X_confidence, self.target_ones)
            discImageLoss_XHat = self.criterion_gen(X_hat_confidence, self.target_zeros)
            discImageLoss_XTilde = self.criterion_gen(X_tilde_confidence, self.target_zeros)
            details_info.log('discImageLoss_X', discImageLoss_X, 2)
            details_info.log('discImageLoss_XHat', discImageLoss_XHat)
            details_info.log('discImageLoss_XTilde', discImageLoss_XTilde)
 
    def calc_gen_losses(self):
        model_files = self.get_model_files()
        for model_file in model_files:
            model = copy.copy(self)
            model.load_state_dict(torch.load(model_file))
            details_info = DetailsInfo()
            for batch, (real_samples, _) in enumerate(self.dataloader):
                real_samples, real_samples_transformed = real_samples
                real_samples = real_samples.to(self.device)
                real_samples_transformed = real_samples_transformed.to(self.device)
                self.calc_discriminators_stats(real_samples, real_samples_transformed, model, details_info)
            print(model_file)
            details_info.printStats(0)
            details_info.printStats(1)

    def train_step_generators(self, X):
        """Train the generator one step and return the loss."""
        self.generator.zero_grad()
        self.encoder.zero_grad()

        Z = self.noise_fn(self.batch_size)

        X_hat = self.generator(Z)
        Z_hat = self.encoder(X)
        X_tilde = self.generator(Z_hat)
        Z_tilde = self.encoder(X_hat)

        X_copy_tilde = torch.cat((X, X_tilde), dim=1)
        X_copy_tilde_rolled = torch.cat((X, torch.roll(X_tilde, 1, 0)), dim=1)

        #self.discriminator_image.eval()
        #self.discriminator_image2.eval()
        #self.discriminator_latent.eval()
        X_copy_tilde_confidence = self.discriminator_image2(X_copy_tilde)
        X_not_copy_tilde_confidence = self.discriminator_image2(X_copy_tilde_rolled)
        X_hat_confidence = self.discriminator_image(X_hat)
        Z_hat_confidence = self.discriminator_latent(Z_hat)
        X_tilde_confidence = self.discriminator_image(X_tilde)
        Z_tilde_confidence = self.discriminator_latent(Z_tilde)
        #self.discriminator_image.train()
        #self.discriminator_image2.train()
        #self.discriminator_latent.train()

        X_copy_tilde_loss = self.criterion_gen(X_copy_tilde_confidence, self.target_ones)
        X_not_copy_tilde_loss = self.criterion_gen(X_not_copy_tilde_confidence, self.target_zeros)
        X_hat_loss = self.criterion_gen(X_hat_confidence, self.target_ones)
        Z_hat_loss = self.criterion_gen(Z_hat_confidence, self.target_ones)
        X_tilde_loss = self.criterion_gen(X_tilde_confidence, self.target_ones)
        Z_tilde_loss = self.criterion_gen(Z_tilde_confidence, self.target_ones)

        X_recon_loss = self.criterion_recon_image(X_tilde, X)
        Z_recon_loss = self.criterion_recon_latent(Z_tilde, Z)

        X_loss = (X_hat_loss + X_tilde_loss) / 2
        Z_loss = (Z_hat_loss + Z_tilde_loss) / 2
        loss_copy = (X_copy_tilde_loss + X_not_copy_tilde_loss) / 2
        loss = loss_copy * ALPHA_COPY_IMAGE + X_loss * ALPHA_DISCRIMINATE_IMAGE + Z_loss * ALPHA_DISCRIMINATE_LATENT + X_recon_loss * ALPHA_RECONSTRUCT_IMAGE + Z_recon_loss * ALPHA_RECONSTRUCT_LATENT

        loss.backward()
        self.optim_e.step()
        self.optim_g.step()

        return X_loss.item(), Z_loss.item(), X_recon_loss.item(), Z_recon_loss.item()

    def train_step_discriminators(self, X, X_transformed, cache):
        """Train the discriminator one step and return the losses."""
        X_copy = torch.cat((X, X_transformed), dim=1)
        X_copy_rolled = torch.cat((X, torch.roll(X_transformed, 1, 0)), dim=1)
        self.discriminator_image.zero_grad()
        self.discriminator_image2.zero_grad()
        self.discriminator_latent.zero_grad()

        Z = self.noise_fn(self.batch_size)

        with torch.no_grad():
            X_hat = self.generator(Z)
            cache.add_images(X_hat)
            Z_hat = self.encoder(X)
            X_tilde = self.generator(Z_hat)
            #cache.add_images(X_tilde)
            Z_tilde = self.encoder(X_hat)
            X_mem = cache.get_images(self.batch_size).to(self.device)
        
        X_copy_tilde = torch.cat((X, X_tilde), dim=1)
        X_copy_tilde_rolled = torch.cat((X, torch.roll(X_tilde, 1, 0)), dim=1)

        X_copy_confidence = self.discriminator_image2(X_copy)
        X_not_copy_confidence = self.discriminator_image2(X_copy_rolled)
        X_copy_tilde_confidence = self.discriminator_image2(X_copy_tilde)
        X_not_copy_tilde_confidence = self.discriminator_image2(X_copy_tilde_rolled)

        X_confidence = self.discriminator_image(X)
        X_hat_confidence = self.discriminator_image(X_hat)
        X_tilde_confidence = self.discriminator_image(X_tilde)
        X_mem_confidence = self.discriminator_image(X_mem)
        Z_confidence = self.discriminator_latent(Z)
        Z_hat_confidence = self.discriminator_latent(Z_hat)
        Z_tilde_confidence = self.discriminator_latent(Z_tilde)

        X_copy_loss = self.criterion_gen(X_copy_confidence, self.target_ones)
        X_not_copy_loss = self.criterion_gen(X_not_copy_confidence, self.target_zeros)
        X_copy_tilde_loss = self.criterion_gen(X_copy_tilde_confidence, self.target_zeros)
        X_not_copy_tilde_loss = self.criterion_gen(X_not_copy_tilde_confidence, self.target_zeros)

        X_loss = self.criterion_gen(X_confidence, self.target_ones)
        X_hat_loss = self.criterion_gen(X_hat_confidence, self.target_zeros)
        X_tilde_loss = self.criterion_gen(X_tilde_confidence, self.target_zeros)
        X_mem_loss = self.criterion_gen(X_mem_confidence, self.target_zeros)
        Z_loss = 2 * self.criterion_gen(Z_confidence, self.target_ones)
        Z_hat_loss = self.criterion_gen(Z_hat_confidence, self.target_zeros)
        Z_tilde_loss = self.criterion_gen(Z_tilde_confidence, self.target_zeros)

        loss_copy = (3*X_copy_loss + X_not_copy_loss + X_copy_tilde_loss + X_not_copy_tilde_loss) / 6
        loss_images = (3*X_loss + X_hat_loss + X_tilde_loss + X_mem_loss) / 6
        loss_latent = (Z_loss + Z_hat_loss + Z_tilde_loss) / 4
        loss = loss_copy + loss_images + loss_latent

        loss.backward()
        self.optim_di2.step()
        self.optim_di.step()
        self.optim_dl.step()

        return loss_images.item(), loss_latent.item(), loss_copy.item(), np.array([X_loss.item(), X_hat_loss.item(), X_tilde_loss.item(), X_mem_loss.item(), X_copy_loss.item(), X_not_copy_loss.item(), X_copy_tilde_loss.item(), X_not_copy_tilde_loss.item()])

    def train_epoch(self, cache):
        """Train both networks for one epoch and return the losses.
        """
        ldx, ldz, lgx, lgz, lrx, lrz = 0, 0, 0, 0, 0, 0
        ldxc = 0
        details = None
        for batch, (real_samples, _) in enumerate(self.dataloader):
            real_samples, real_samples_transformed = real_samples
            real_samples = real_samples.to(self.device)
            real_samples_transformed = real_samples_transformed.to(self.device)
            ldx_, ldz_, ldxc_, details_ = self.train_step_discriminators(real_samples, real_samples_transformed, cache)
            ldx += ldx_
            ldz += ldz_
            ldxc += ldxc_
            if details is None:
                details = details_
            else:
                details += details_
            lgx_, lgz_, lrx_, lrz_ = self.train_step_generators(real_samples)
            lgx += lgx_
            lgz += lgz_
            lrx += lrx_
            lrz += lrz_

        n = len(self.dataloader)
        lgx /= n
        lgz /= n
        ldx /= n
        ldz /= n
        lrx /= n
        lrz /= n

        ldxc /= n
        details /= n

        print(f"Gx={lgx:.4f}, Gz={lgz:.4f}, Dx={ldx:.3f}, Dz={ldz:.3f} Rx={lrx:.3f} Rz={lrz:.3f} DxCopy={ldxc:.3f}")
        print(details)

