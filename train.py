import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from matplotlib import pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
import numpy as np
from dataset import MY_MNIST_Train
from module import DnCNN

# field = np.random.normal(0, 1, [iteration, image_size, image_size])
# field = torch.from_numpy(field).cuda()

train_dataset = MY_MNIST_Train(root1=r'C:\Users\chenda\Desktop\Mnist_1on1_dncnn_2000_100_128/train-pt/1.pt',
                               root2=r'C:\Users\chenda\Desktop\Mnist_1on1_dncnn_2000_100_128/train-GI-pt/1.pt')

print('done')
# plt.imshow(train_dataset[0][0][0])
# plt.show()
torch.cuda.empty_cache()
batch_size = 16  # 16
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 加载数据集
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
model = DnCNN().cuda()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=2e-5, weight_decay=3e-4)

model = model.to(device)
criterion = criterion.to(device)

# 训练
for epoch in range(1, 105):
    print("-----------------第{}轮训练开始---------------------".format(epoch))
    model.train()  # 作用是启用batch normalization和drop out
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()  # 把梯度置零

        data = data.to(device)
        target = target.to(device)
        # print(data.shape)
        # print(target.shape)
        data = data.to(torch.float32)
        output = model(data)
        # print(output.shape)
        # print(target.shape)

        output = output.to(torch.float32)  # 将数据类型改变
        target = target.to(torch.float32)
        # print(output[0])
        # print(target[0])
        loss = criterion(output, target)  # 计算损失
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('训练次数: {}，[{}/{} ({:.0f}%)]\t Loss: {}'.format(
                epoch, batch_idx * len(data.cuda()), len(train_loader.dataset), 100. * batch_idx / len(train_loader),
                loss.item()))
    if epoch % 50 == 0:
        torch.save(model, "./model_weight/module_{}.pth".format(epoch))  # 每十轮保留一次参数
        print("第{}轮数据已保存".format(epoch))
