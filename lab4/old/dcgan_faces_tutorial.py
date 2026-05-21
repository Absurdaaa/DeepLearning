#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 如果你在 Google Colab 中运行 notebook，可以参考下面这个官方说明：
# https://pytorch.org/tutorials/beginner/colab
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#%matplotlib inline
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

# 设置随机种子，保证结果可复现
manualSeed = 999
#manualSeed = random.randint(1, 10000) # 如果你希望得到新的随机结果，可以启用这一行
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
torch.use_deterministic_algorithms(True) # 为了可复现结果，启用确定性算法


# In[ ]:


# 数据集根目录
dataroot = "data/celeba"

# DataLoader 的工作线程数
workers = 2

# 训练时的 batch size
batch_size = 128

# 训练图像的空间尺寸，所有图像都会先被变换到这个大小。
image_size = 64

# 训练图像的通道数，彩色图像通常为 3
nc = 3

# 潜变量 z 的维度（也就是生成器输入维度）
nz = 100

# 生成器特征图通道宽度
ngf = 64

# 判别器特征图通道宽度
ndf = 64

# 训练 epoch 数
num_epochs = 5

# 优化器学习率
lr = 0.0002

# Adam 优化器的 beta1 超参数
beta1 = 0.5

# 可用 GPU 数量；设为 0 时使用 CPU。
ngpu = 1


# In[ ]:


# 按当前目录结构，可以直接使用 `ImageFolder` 读取数据。
# 创建数据集
dataset = dset.ImageFolder(root=dataroot,
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
# 创建 dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=workers)

# 选择训练设备
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

# 可视化一部分训练图像
real_batch = next(iter(dataloader))
plt.figure(figsize=(8,8))
plt.axis("off")
plt.title("Training Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
plt.show()


# In[ ]:


# 给 `netG` 和 `netD` 使用的自定义权重初始化函数
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# In[ ]:


# 生成器代码

class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # 输入是潜变量 Z，先进入转置卷积层
            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # 当前特征图尺寸：`(ngf*8) x 4 x 4`
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # 当前特征图尺寸：`(ngf*4) x 8 x 8`
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # 当前特征图尺寸：`(ngf*2) x 16 x 16`
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # 当前特征图尺寸：`(ngf) x 32 x 32`
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # 当前特征图尺寸：`(nc) x 64 x 64`
        )

    def forward(self, input):
        return self.main(input)


# In[ ]:


# 创建生成器
netG = Generator(ngpu).to(device)

# 如果需要，可以启用多 GPU
if (device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))

# 应用 `weights_init`，把权重初始化为 `mean=0`、`stdev=0.02` 的正态分布。
netG.apply(weights_init)

# 打印模型结构
print(netG)


# In[ ]:


class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # 输入尺寸：`(nc) x 64 x 64`
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 当前特征图尺寸：`(ndf) x 32 x 32`
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # 当前特征图尺寸：`(ndf*2) x 16 x 16`
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # 当前特征图尺寸：`(ndf*4) x 8 x 8`
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # 当前特征图尺寸：`(ndf*8) x 4 x 4`
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)


# In[ ]:


# 创建判别器
netD = Discriminator(ngpu).to(device)

# 如果需要，可以启用多 GPU
if (device.type == 'cuda') and (ngpu > 1):
    netD = nn.DataParallel(netD, list(range(ngpu)))
    
# 应用 `weights_init` 初始化权重。
netD.apply(weights_init)

# 打印模型结构
print(netD)


# In[ ]:


# 初始化 `BCELoss`
criterion = nn.BCELoss()

# 创建一批固定潜变量，用于可视化生成器训练过程
fixed_noise = torch.randn(64, nz, 1, 1, device=device)

# 约定训练时真实 / 伪造样本的标签
real_label = 1.
fake_label = 0.

# 为 G 和 D 分别设置 Adam 优化器
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))


# In[ ]:


# 训练循环

# 用于记录训练过程的列表
img_list = []
G_losses = []
D_losses = []
iters = 0

print("Starting Training Loop...")
# 遍历每个 epoch
for epoch in range(num_epochs):
    # 遍历 dataloader 中的每个 batch
    for i, data in enumerate(dataloader, 0):
        
        ############################
        # (1) 更新判别器 D：最大化 log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## 使用全真实样本 batch 训练
        netD.zero_grad()
        # 整理当前 batch
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
        # 把真实样本送入 D 做前向传播
        output = netD(real_cpu).view(-1)
        # 计算全真实样本 batch 上的损失
        errD_real = criterion(output, label)
        # 反向传播，计算 D 的梯度
        errD_real.backward()
        D_x = output.mean().item()

        ## 使用全伪造样本 batch 训练
        # 生成一批潜变量
        noise = torch.randn(b_size, nz, 1, 1, device=device)
        # 用 G 生成一批伪造图像
        fake = netG(noise)
        label.fill_(fake_label)
        # 把伪造图像送入 D 做分类
        output = netD(fake.detach()).view(-1)
        # 计算全伪造样本 batch 上的 D 损失
        errD_fake = criterion(output, label)
        # 计算这一批的梯度，并和前面的真实样本梯度累积
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # 把真实样本损失和伪造样本损失相加，得到 D 的总损失
        errD = errD_real + errD_fake
        # 更新 D
        optimizerD.step()

        ############################
        # (2) 更新生成器 G：最大化 log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.fill_(real_label)  # 对生成器来说，希望伪造样本被判成真实样本
        # 因为 D 已经更新过，这里要重新把伪造样本送入 D
        output = netD(fake).view(-1)
        # 基于当前输出计算 G 的损失
        errG = criterion(output, label)
        # 计算 G 的梯度
        errG.backward()
        D_G_z2 = output.mean().item()
        # 更新 G
        optimizerG.step()
        
        # 输出训练统计信息
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
        
        # 记录损失，便于后续画图
        G_losses.append(errG.item())
        D_losses.append(errD.item())
        
        # 定期保存固定噪声对应的生成结果，观察生成器学习进度
        if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
            
        iters += 1


# In[ ]:


plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses,label="G")
plt.plot(D_losses,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()


# In[ ]:


fig = plt.figure(figsize=(8,8))
plt.axis("off")
ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)

HTML(ani.to_jshtml())


# In[ ]:


# 从 dataloader 中取一批真实图像
real_batch = next(iter(dataloader))

# 绘制真实图像
plt.figure(figsize=(15,15))
plt.subplot(1,2,1)
plt.axis("off")
plt.title("Real Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))

# 绘制最后一个 epoch 生成的图像
plt.subplot(1,2,2)
plt.axis("off")
plt.title("Fake Images")
plt.imshow(np.transpose(img_list[-1],(1,2,0)))
plt.show()
