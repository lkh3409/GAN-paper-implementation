import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import torchvision
import torchvision.utils as utils
import torchvision.datasets as dsets
import torchvision.transforms as transforms

import random

import time

# Set random seed for reproducibility
manualSeed = 2020
#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

is_cuda = torch.cuda.is_available()
device = torch.device('cuda' if is_cuda else 'cpu')


# hyper-parameter
batch_size = 64
# standardizer
standardizer = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=0.5,
                                                        std=0.5)])
# MNIST dataset
train_data = dsets.MNIST(root='../data/', train=True, transform=standardizer, download=True)
# Data loader
train_data_loader = torch.utils.data.DataLoader(train_data, batch_size, shuffle=True)
# function for visualization
def tc_imshow(img, lbl=""):
    if img.size(0) == 1:
        plt.imshow(img.squeeze(), cmap='gray')
    else:
        plt.imshow(np.transpose(img, (1, 2, 0)))
        
    plt.title(lbl)
    plt.axis('off')
# visualize
mini_batch_img, mini_batch_lbl = next(iter(train_data_loader))

plt.figure(figsize=(4,5))
for i in range(16):
    plt.subplot(4,4,i+1)
    tc_imshow(img=mini_batch_img[i] /2+0.5 ,
              lbl=train_data.classes[mini_batch_lbl[i].numpy()])
    plt.axis('off')
    
plt.savefig('../result/GAN/1-GAN/1-dataloader-example.png', dpi=300)

# hyper-parameter
dim_noise = 100
# Random sampling from normal distribution
def random_sample_z_space(batch_size=1, dim_noise=100):
    return torch.randn(batch_size, dim_noise, device=device)
# Random sampling
z = random_sample_z_space(batch_size)


# hyper-paremeters
dim_hidden = 256
sz_output = 28
num_channels = 1
dim_output = sz_output**2
img_shape = (num_channels, sz_output, sz_output)
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.model = nn.Sequential(
                    nn.Linear(dim_noise, dim_hidden),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(dim_hidden, dim_hidden),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(dim_hidden, dim_output),
                    nn.Tanh()
        )
    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *img_shape)
        return img
G = Generator().to(device)

# visualize
utils.save_image(G(z)[:25].cpu().detach(), "../result/GAN/1-GAN/2-G(z).png", nrow=5, normalize=True)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(dim_output, dim_hidden),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.Linear(dim_hidden, dim_hidden),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.Linear(dim_hidden, 1),
            nn.Sigmoid()    
        )
    
    def forward(self, img):
        flat_img = img.view(img.size(0), -1)
        check_validity = self.model(flat_img)
        
        return check_validity
D = Discriminator().to(device)


generator = Generator().to(device)
discriminator = Discriminator().to(device)

# hyper-parameters
learning_rate = 0.0002
beta1 = 0.9
adversarial_loss = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=learning_rate, betas=(beta1, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(beta1, 0.999))

# 시작 시간 기록
start_time = time.time()

# hyper-parameters
num_epochs = 200
interval_save_img = 5000

Tensor = torch.cuda.FloatTensor if is_cuda else torch.FloatTensor
losses = []

for idx_epoch in range(num_epochs):
    for idx_batch, (imgs, _) in enumerate(train_data_loader):
        # Ground truth variables indicating real/fake
        real_ground_truth = Variable(Tensor(imgs.size(0), 1).fill_(1.0), requires_grad=False)
        fake_ground_truth = Variable(Tensor(imgs.size(0), 1).fill_(0.0), requires_grad=False) 
        
        # Real image
        real_imgs = Variable(imgs.type(Tensor))
                
        #####################
        # Train Generator
        
        optimizer_G.zero_grad()
        
        # Random sample noise
        z = random_sample_z_space(imgs.size(0))
                
        # Generate image
        gen_imgs = generator(z)
        
        # Generator's loss: loss between D(G(z)) and real ground truth
        loss_G = adversarial_loss(discriminator(gen_imgs), real_ground_truth)
        
        loss_G.backward()
        optimizer_G.step()
        
        
        #####################
        # Train Discriminator
        
        optimizer_D.zero_grad()
        
        loss_real = adversarial_loss(discriminator(real_imgs), real_ground_truth)
        loss_fake = adversarial_loss(discriminator(gen_imgs.detach()), fake_ground_truth)
        loss_D = (loss_real+loss_fake)/2
        
        loss_D.backward()
        optimizer_D.step()
        
        
        #####################
        # archieve loss
        losses.append([loss_G.item(), loss_D.item()])
        
        # Print progress
        if idx_batch % 938 == 0:
                        print("[Epoch {}/{}] [Batch {}/{}] loss_G: {:.6f}, loss_D: {:.6f}".format(idx_epoch, num_epochs,
                                                                                      idx_batch, len(train_data_loader),
                                                                                      loss_G, loss_D))
                    
        batches_done = idx_epoch * len(train_data_loader) + idx_batch
        if batches_done % interval_save_img == 0:
            utils.save_image(gen_imgs.data[:25], "../result/GAN/1-GAN/3-{}.png".format(batches_done), nrow=5, normalize=True)

# 종료 시간 기록
end_time = time.time()

# 소요 시간 계산
training_time = end_time - start_time

# 시, 분, 초로 변환
minutes = int(training_time // 60)
seconds = int(training_time % 60)
hours = int(minutes // 60)
minutes = int(minutes % 60)

# 결과 출력
print(f"훈련 소요 시간: {hours}시간 {minutes}분 {seconds}초")

 # Random sample noise
z = random_sample_z_space(batch_size)

# Generate image
gen_imgs = generator(z)

# visualize
utils.save_image(gen_imgs.data[:25].cpu().detach(), "../result/GAN/1-GAN/4-fake-images.png", nrow=5, normalize=True)

torch.save({
    'epoch': num_epochs,
    'model_G_state_dict': generator.state_dict(),
    'model_D_state_dict': discriminator.state_dict(),
    'optimizer_G_state_dict': optimizer_G.state_dict(),
    'optimizer_D_state_dict': optimizer_D.state_dict(),
    'losses': losses
}, '../result/GAN/1-GAN/model_weights.pth')

z_opposites = random_sample_z_space(2)
fake_img = generator(z_opposites)

plt.subplot(121); tc_imshow(img=fake_img[0].cpu().detach() /2+0.5  )
plt.subplot(122); tc_imshow(img=fake_img[1].cpu().detach() /2+0.5  )

plt.savefig('../result/GAN/1-GAN/5-side-images.png', dpi=300)

num_interpolation = 10

z_interpolation = Variable(Tensor(np.linspace(z_opposites[0].cpu(), z_opposites[1].cpu(), num_interpolation)))
fake_img = generator(z_interpolation)

plt.figure(figsize=(12,2))
for i in range(num_interpolation):
    plt.subplot(1,num_interpolation,i+1)
    tc_imshow(img=fake_img[i].cpu().detach() /2+0.5 )
    
plt.savefig('../result/GAN/1-GAN/6-interpolation.png', dpi=300)