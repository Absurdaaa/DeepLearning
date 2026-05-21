#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
print(sys.version) # Python 3.6
import torch
import torch.nn as nn
import torchvision.datasets
import torchvision.transforms as transforms
import torch.nn.functional as F
import torchvision.utils as vutils
print(torch.__version__) # PyTorch 1.0.1

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

def show_imgs(x, new_fig=True):
    grid = vutils.make_grid(x.detach().cpu(), nrow=8, normalize=True, pad_value=0.3)
    grid = grid.transpose(0,2).transpose(0,1) # 把通道维移到最后，便于 matplotlib 显示
    if new_fig:
        plt.figure()
    plt.imshow(grid.numpy())


# In[2]:


class Discriminator(torch.nn.Module):
    def __init__(self, inp_dim=784):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(inp_dim, 128)
        self.nonlin1 = nn.LeakyReLU(0.2)
        self.fc2 = nn.Linear(128, 1)
    def forward(self, x):
        x = x.view(x.size(0), 784) # 拉平成向量：(bs x 1 x 28 x 28) -> (bs x 784)
        h = self.nonlin1(self.fc1(x))
        out = self.fc2(h)
        out = torch.sigmoid(out)
        return out


# In[3]:


class Generator(nn.Module):
    def __init__(self, z_dim=100):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(z_dim, 128)
        self.nonlin1 = nn.LeakyReLU(0.2)
        self.fc2 = nn.Linear(128, 784)
    def forward(self, x):
        h = self.nonlin1(self.fc1(x))
        out = self.fc2(h)
        out = torch.tanh(out) # 输出范围压到 [-1, 1]
        # 重排成图像张量
        out = out.view(out.size(0), 1, 28, 28)
        return out


# In[4]:


# 按照类定义实例化生成器和判别器。
D = Discriminator()
print(D)
G = Generator()
print(G)


# In[5]:


# 下载 FashionMNIST 数据；如果本地已经下载过，可以把路径改到已有文件位置。
# dataset = torchvision.datasets.MNIST(root='./MNISTdata', ...)
dataset = torchvision.datasets.FashionMNIST(root='./FashionMNIST/',
                       transform=transforms.Compose([transforms.ToTensor(),
                                                     transforms.Normalize((0.5,), (0.5,))]),
                       download=True)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)


# In[6]:


ix=149
x, _ = dataset[ix]
plt.matshow(x.squeeze().numpy(), cmap=plt.cm.gray)
plt.colorbar()


# In[7]:


# 单张图像送入判别器：
Dscore = D(x)
Dscore


# In[9]:


# 演示如何从 dataloader 中取出一个 batch：
xbatch, _ = next(iter(dataloader)) # 64 x 1 x 28 x 28：一个包含 64 个样本的 mini-batch
xbatch.shape
D(xbatch) # 64x1 张量：表示 64 个样本分别为真实图像的概率
D(xbatch).shape


# In[10]:


show_imgs(xbatch)


# In[11]:


x = torch.randn(2,2, requires_grad=True)
x


# In[12]:


# 此时还没有梯度：
print(x.grad)


# In[13]:


y=(x**2 + x)
z = y.sum()
z


# In[14]:


z.backward()
x.grad


# In[15]:


2*x+1


# In[16]:


for p in G.parameters():
    print(p.grad)


# In[17]:


torch.manual_seed(23231)
x1 = torch.Tensor([1, 2, 3, -3, -2])
y = torch.Tensor ([3, 6, 9, -9, -6]).view(5,1)
x2 = torch.randn(5)
x = torch.stack([x1, x2], dim=1) # 5 x 2 的输入，共 5 个样本，每个样本 2 维
# theta = torch.randn(1,2, requires_grad=True) # 与下面这行大致等价：
theta = torch.nn.Parameter(torch.randn(1,2))
# 从随机初始化开始，梯度会告诉我们应该往哪个方向更新。
print('x:\n', x)
print('y:\n', y)
print('theta at random initialization: ', theta)
thetatrace = [theta.data.clone()] # 记录初始值，便于后面画轨迹


# In[18]:


ypred = x @ theta.t() # 矩阵乘法：(N x 2) * (2 x 1) -> (N x 1)
print('ypred:\n', ypred)
loss = ((ypred-y)**2).mean() # 均方误差 MSE
print('mse loss: ', loss.item())
loss.backward()
print('dL / d theta:\n', theta.grad)
# 沿负梯度方向更新参数
theta.data.add_(-0.1 * theta.grad.data)
# 更新后把梯度清零
theta.grad.zero_()
print('theta:\n', theta)
thetatrace.append(theta.data.clone()) # 继续记录参数轨迹


# In[19]:


# 把 SGD 过程中参数 theta 的移动轨迹画到二维平面上，红点是真实解。
thetas = torch.cat(thetatrace, dim=0).numpy()
plt.figure()
plt.plot(thetas[:,0], thetas[:, 1], 'x-')
plt.plot(3, 0, 'ro')
plt.xlabel('theta[0]')
plt.ylabel('theta[1]')


# In[20]:


torch.manual_seed(23801)
net = nn.Linear(2,1, bias=False)
optimizer = torch.optim.SGD(net.parameters(), lr=0.1) # 用 `optimizer.step()` 执行参数更新
# 这里直接复用上面定义的 x、y。真实任务里通常会从 dataloader 里不断取不同的 mini-batch。
for i in range(100): # 做 100 次梯度下降更新
    ypred = net(x)
    loss = ((ypred-y)**2).mean() # 均方误差 MSE
    optimizer.zero_grad()
    loss.backward()
    # 这里不再手动写 W.data -= lr * W.grad，而是交给优化器完成
    optimizer.step()
print(net.weight)


# In[21]:


# 重新实例化一次判别器和生成器：
D = Discriminator()
print(D)
G = Generator()
print(G)
# 然后配置优化器
optimizerD = torch.optim.SGD(D.parameters(), lr=0.01)
optimizerG = torch.optim.SGD(G.parameters(), lr=0.01)


# In[22]:


# 使用 BCE 损失来对应上面的目标函数：
criterion = nn.BCELoss()


# In[24]:


# 第 1 步：更新判别器
x_real, _ = next(iter(dataloader))
lab_real = torch.ones(64, 1)
lab_fake = torch.zeros(64, 1)
# 清空上一轮累计的梯度
optimizerD.zero_grad()

D_x = D(x_real)
lossD_real = criterion(D_x, lab_real)

z = torch.randn(64, 100) # 随机噪声，64 个样本，z_dim=100
x_gen = G(z).detach()
D_G_z = D(x_gen)
lossD_fake = criterion(D_G_z, lab_fake)

lossD = lossD_real + lossD_fake
lossD.backward()
optimizerD.step()

# print(D_x.mean().item(), D_G_z.mean().item())


# In[25]:


# 第 2 步：更新生成器
# 注意损失里只有涉及 G 的那一项会影响生成器参数。
# 清空上一轮累计的梯度
optimizerG.zero_grad()

z = torch.randn(64, 100) # 随机噪声，64 个样本，z_dim=100
D_G_z = D(G(z))
lossG = criterion(D_G_z, lab_real) # 对应 -log D(G(z))

lossG.backward()
optimizerG.step()

print(D_G_z.mean().item())


# In[ ]:


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print('Device: ', device)
# 重新初始化 D 和 G
D = Discriminator().to(device)
G = Generator().to(device)
# 重新设置优化器（GAN 任务里通常 Adam 会比纯 SGD 更稳）
optimizerD = torch.optim.SGD(D.parameters(), lr=0.03)
optimizerG = torch.optim.SGD(G.parameters(), lr=0.03)
# optimizerD = torch.optim.Adam(D.parameters(), lr=0.0002)
# optimizerG = torch.optim.Adam(G.parameters(), lr=0.0002)
lab_real = torch.ones(64, 1, device=device)
lab_fake = torch.zeros(64, 1, device=device)


# 用于可视化和日志记录
collect_x_gen = []
fixed_noise = torch.randn(64, 100, device=device)
fig = plt.figure() # 复用这一张图做动态更新
plt.ion()

for epoch in range(3): # 训练 3 个 epoch
    for i, data in enumerate(dataloader, 0):
        # 第 1 步：更新判别器
        x_real, _ = next(iter(dataloader))
        x_real = x_real.to(device)
        # 清空上一轮累计梯度
        optimizerD.zero_grad()

        D_x = D(x_real)
        lossD_real = criterion(D_x, lab_real)

        z = torch.randn(64, 100, device=device) # 随机噪声，64 个样本，z_dim=100
        x_gen = G(z).detach()
        D_G_z = D(x_gen)
        lossD_fake = criterion(D_G_z, lab_fake)

        lossD = lossD_real + lossD_fake
        lossD.backward()
        optimizerD.step()
        
        # 第 2 步：更新生成器
        # 清空上一轮累计梯度
        optimizerG.zero_grad()

        z = torch.randn(64, 100, device=device) # 随机噪声，64 个样本，z_dim=100
        x_gen = G(z)
        D_G_z = D(x_gen)
        lossG = criterion(D_G_z, lab_real) # 对应 -log D(G(z))

        lossG.backward()
        optimizerG.step()
        if i % 100 == 0:
            x_gen = G(fixed_noise)
            show_imgs(x_gen, new_fig=False)
            fig.canvas.draw()
            print('e{}.i{}/{} last mb D(x)={:.4f} D(G(z))={:.4f}'.format(
                epoch, i, len(dataloader), D_x.mean().item(), D_G_z.mean().item()))
    # 一个 epoch 结束后保存一份固定噪声的生成结果
    x_gen = G(fixed_noise)
    collect_x_gen.append(x_gen.detach().clone())


# In[25]:


for x_gen in collect_x_gen:
    show_imgs(x_gen)


# In[ ]:


