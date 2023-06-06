import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torchvision.transforms import Compose, ToTensor, Lambda, ToPILImage, CenterCrop, Resize
from torchvision import datasets, transforms
from tqdm import tqdm
import numpy as np
from pathlib import Path
import os

from ex02_model import Unet
from ex02_diffusion import Diffusion, linear_beta_schedule
from torchvision.utils import save_image

import argparse

torch.manual_seed(0)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train a neural network to diffuse images')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='input batch size for training (default: 64)')
    parser.add_argument('--timesteps', type=int, default=100,
                        help='number of timesteps for diffusion model (default: 100)')
    parser.add_argument('--epochs', type=int, default=5,
                        help='number of epochs to train (default: 5)')
    parser.add_argument('--lr', type=float, default=0.003,
                        help='learning rate (default: 0.003)')
    # parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', action='store_true',
                        default=False, help='disables CUDA training')
    # parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--log_interval', type=int, default=100,
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save_model', action='store_true',
                        default=False, help='For Saving the current Model')
    parser.add_argument('--run_name', type=str, default="DDPM")
    parser.add_argument('--dry_run', action='store_true',
                        default=False, help='quickly check a single pass')
    return parser.parse_args()


def sample_and_save_images(n_images, diffusor, model, device, store_path):
    # TODO: Implement - adapt code and method signature as needed
    imgs = diffusor.sample(model, torch.tensor([1]).to(device), 128, n_images, 3)
    # print([img.shape for img in imgs])
    for i, img in enumerate(imgs):
        save_image(img, f"{store_path}/{i}.jpg")
    
def val(model, testloader, diffusor, epoch, device, args):
    timesteps = args.timesteps

    pbar = tqdm(testloader)
    for step, (images, labels) in enumerate(pbar):

        images = images.to(device)

        # Algorithm 1 line 3: sample t uniformly for every example in the batch
        t = torch.randint(0, timesteps, (len(images),), device=device).long()
        loss = diffusor.p_losses(
            model, images, t, labels.to(device), loss_type="l2")

        if step % args.log_interval == 0:
            print('Val Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, step * len(images), len(testloader.dataset),
                100. * step / len(testloader), loss.item()))
        if args.dry_run:
            break


def train(model, trainloader, optimizer, diffusor, epoch, device, args):
    batch_size = args.batch_size
    timesteps = args.timesteps

    pbar = tqdm(trainloader)
    for step, (images, labels) in enumerate(pbar):

        images = images.to(device)
        optimizer.zero_grad()

        # Algorithm 1 line 3: sample t uniformly for every example in the batch
        # classes = torch.randint(0, num_classes, (len(images), )).to(device)
        t = torch.randint(0, timesteps, (len(images),), device=device).long()
        loss = diffusor.p_losses(
            model, images, t, labels.to(device), loss_type="l2")

        loss.backward()
        optimizer.step()

        if step % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, step * len(images), len(trainloader.dataset),
                100. * step / len(trainloader), loss.item()))
        if args.dry_run:
            break


def test(model, testloader, diffusor, device, args):
    # TODO (2.2): implement testing functionality, including generation of stored images.
    timesteps = args.timesteps

    pbar = tqdm(testloader)
    for step, (images, labels) in enumerate(pbar):

        images = images.to(device)

        # Algorithm 1 line 3: sample t uniformly for every example in the batch
        t = torch.randint(0, timesteps, (len(images),), device=device).long()
        loss = diffusor.p_losses(
            model, images, t, labels.to(device), loss_type="l2")

        if step % args.log_interval == 0:
            print('Test: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                 step * len(images), len(testloader.dataset),
                100. * step / len(testloader), loss.item()))
        if args.dry_run:
            break



def run(args):
    timesteps = args.timesteps
    image_size = 32  # TODO (2.5): Adapt to new dataset
    channels = 3
    epochs = args.epochs
    batch_size = args.batch_size
    device = "cuda" if not args.no_cuda and torch.cuda.is_available() else "cpu"

    # define image transformations (e.g. using torchvision)
    transform = Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),    # turn into torch Tensor of shape CHW, divide by 255
        # scale data to [-1, 1] to aid diffusion process
        transforms.Lambda(lambda t: (t * 2) - 1)
    ])

    global reverse_transform
    reverse_transform = Compose([
        Lambda(lambda t: (t.clamp(-1, 1) + 1) / 2),
        Lambda(lambda t: t.permute(1, 2, 0)),  # CHW to HWC
        Lambda(lambda t: t * 255.),
        Lambda(lambda t: t.numpy().astype(np.uint8)),
        ToPILImage(),
    ])

    dataset = datasets.CIFAR10(
        '/proj/aimi-adl/CIFAR10/', download=True, train=True, transform=transform)
    trainset, valset = torch.utils.data.random_split(
        dataset, [int(len(dataset) * 0.9), len(dataset) - int(len(dataset) * 0.9)])
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(valset, batch_size=batch_size, shuffle=False)

    # Download and load the test data
    testset = datasets.CIFAR10(
        '/proj/aimi-adl/CIFAR10/', download=True, train=False, transform=transform)
    testloader = DataLoader(
        testset, batch_size=int(batch_size/2), shuffle=True)

    global num_classes
    print(dataset.classes)
    num_classes = len(dataset.classes)

    model = Unet(dim=image_size, channels=channels, dim_mults=(
        1, 2, 4,), class_free_guidance=True, num_classes=num_classes, p_uncond=0.3).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr)
    def my_scheduler(x): return linear_beta_schedule(0.0001, 0.02, x)
    diffusor = Diffusion(timesteps, my_scheduler, image_size, device)
    # for epoch in range(epochs):
    #     train(model, trainloader, optimizer, diffusor, epoch, device, args)
    #     val(model, valloader, diffusor, epoch, device, args)

    # test(model, testloader, diffusor, device, args)

    save_path = "./output"  
    n_images = 8
    model.load_state_dict(torch.load("./model.pt"))
    sample_and_save_images(n_images, diffusor, model, device, save_path)
    # torch.save(model.state_dict(), "./model.pt")


# source: https://huggingface.co/blog/annotated-diffusion
def get_noisy_image(diffusor, x_start, t):
    # add noise
    x_noisy = diffusor.q_sample(x_start, t)

    # turn back into PIL image
    reverse_transform = Compose([
        Lambda(lambda t: (t.clamp(-1, 1) + 1) / 2),
        Lambda(lambda t: t.permute(1, 2, 0)),  # CHW to HWC
        Lambda(lambda t: t * 255.),
        Lambda(lambda t: t.numpy().astype(np.uint8)),
        ToPILImage(),
    ])
    noisy_image = reverse_transform(x_noisy.squeeze().cpu())

    return noisy_image


# source: https://pytorch.org/vision/stable/auto_examples/plot_transforms.html#sphx-glr-auto-examples-plot-transforms-py


def plot(imgs, og_image, with_orig=True, row_title=None, **imshow_kwargs):
    if not isinstance(imgs[0], list):
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]

    num_rows = len(imgs)
    num_cols = len(imgs[0]) + with_orig
    fig, axs = plt.subplots(
        figsize=(200, 200), nrows=num_rows, ncols=num_cols, squeeze=False)
    for row_idx, row in enumerate(imgs):
        row = [og_image] + row if with_orig else row
        for col_idx, img in enumerate(row):
            ax = axs[row_idx, col_idx]
            ax.imshow(np.asarray(img), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    if with_orig:
        axs[0, 0].set(title='Original image')
        axs[0, 0].title.set_size(8)
    if row_title is not None:
        for row_idx in range(num_rows):
            axs[row_idx, 0].set(ylabel=row_title[row_idx])

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    args = parse_args()
    # TODO (2.2): Add visualization capabilities
    run(args)

    # from PIL import Image
    # import requests

    # timesteps = 300
    # image_size = 32  # TODO (2.5): Adapt to new dataset
    # channels = 3
    # epochs = args.epochs
    # batch_size = args.batch_size
    # device = "cuda" if not args.no_cuda and torch.cuda.is_available() else "cpu"

    # image_size = 128
    # def my_scheduler(x): return linear_beta_schedule(0.0001, 0.02, x)
    # diffusor = Diffusion(timesteps, my_scheduler, image_size, device)


    # url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    # image = Image.open(requests.get(url, stream=True).raw) # PIL image of shape HWC
    # transform = Compose([
    #     Resize(image_size),
    #     CenterCrop(image_size),
    #     ToTensor(), # turn into torch Tensor of shape CHW, divide by 255
    #     Lambda(lambda t: (t * 2) - 1),
        
    # ])

    # x_start = transform(image).unsqueeze(0).to(device)
    # plot([get_noisy_image(diffusor, x_start, torch.tensor([t])) for t in [0, 50, 100, 150, 200, 250, 299]], image)

