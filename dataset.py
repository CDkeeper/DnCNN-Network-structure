import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
from torchvision.transforms import functional as F
image_size = 64  # 128
iteration = 400  # 实验室电脑内存极限大致是 450*60000*28*28

# field = np.random.normal(0, 1, [iteration, image_size, image_size])
# field = torch.from_numpy(field).cuda()


class MY_MNIST_Train(Dataset):

    def __init__(self, root1, root2, transform=None):
        self.transform = transform
        pre_data = torch.load(root2)  # 标签就是原图片，pre__data是需要GI图，但这里先读取原图，之后经过gi处理
        self.targets = torch.load(root1)  # 标签就是原图片，pre__data也是原图片但要经过鬼成像处理
        # print("原始数据为{}".format(pre_data[0][0][0]))

        self.data = pre_data
        # batch_ci = 30  # 60000 数据总量为batch_ci和batch_size的积
        # batch_size = 20
        # self.data = torch.zeros((batch_ci * batch_size, image_size, image_size))  # 60000
        #
        # for num in range(batch_ci):
        #     if num % 5000 == 0:
        #         print("Num = %d" % num)
        #     torch.cuda.empty_cache()
        #
        #     # 初始化 B 数组
        #     B = torch.zeros((iteration, batch_size)).cuda()
        #     BI = torch.zeros((batch_size, image_size, image_size)).cuda()
        #
        #     # [500, 28, 28]的图像数组
        #     img_batch = torch.zeros((batch_size, image_size, image_size))
        #
        #     # 读取图片
        #     for i in range(batch_size):
        #         x = pre_data[i + batch_size * num]
        #         transform1 = transforms.ToPILImage(mode="L")
        #         y = transform1(np.uint8(x.numpy()))
        #         y = y.resize((image_size, image_size))
        #         transform2 = transforms.ToTensor()
        #         z = transform2(y)
        #         img_batch[i] = z[0]
        #     img_batch = img_batch.cuda()
        #
        #     # [512]
        #     sum_I1 = torch.zeros(batch_size).cuda()
        #     # [512, 128, 128]
        #     sum_I2 = torch.zeros(img_batch.shape).cuda()
        #     # [512, 128, 128]
        #     sum_ans = torch.zeros(img_batch.shape).cuda()
        #     sum_I2 = sum_I2.permute(1, 2, 0)
        #     sum_ans = sum_ans.permute(1, 2, 0)
        #
        #     field = np.random.normal(0, 1, [iteration, image_size, image_size])
        #     field = torch.from_numpy(field).cuda()
        #
        #     for k in range(iteration):
        #         temp = field[k].repeat(batch_size, 1, 1).cuda()
        #         B[k] = torch.mul(temp, img_batch).sum(2).sum(1)
        #         temp = temp.permute(1, 2, 0)
        #         # 桶测量值求和
        #         sum_I1 = sum_I1 + B[k]
        #         # 热光矩阵求和
        #         sum_I2 = sum_I2 + temp
        #         # 桶测量值乘热光矩阵求和
        #         sum_ans = sum_ans + temp * B[k]
        #
        #     sum_I1 = sum_I1 / iteration
        #     sum_I2 = sum_I2 / iteration
        #     sum_ans = sum_ans / iteration
        #     # [512, 128, 128]
        #     ans = sum_ans - sum_I1 * sum_I2
        #     ans = ans.permute(2, 0, 1)
        #     for i in range(batch_size):
        #         self.data[i + batch_size * num] = ans[i]

    #             for i in range(batch_size):
    #                 x = img_batch[i]
    #                 self.data[i + batch_size * num][0] = x

    #     print(self.data.shape)  #展示某张图对于第几帧的效果图，例如如下为第122张图片的第125帧
    #     To_PIL = transforms.ToPILImage()
    #     img = To_PIL(self.data[122][125])
    #     img.show()

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = img.unsqueeze(0)
        target = target.unsqueeze(0)
        return img, target

    def __len__(self):
        return len(self.data)


class MY_MNIST_Test(Dataset):

    def __init__(self, root, transform=None):
        self.transform = transform
        pre_data = torch.load(root)  # 标签就是原图片，pre__data是需要GI图，但这里先读取原图，之后经过gi处理
        old_data = torch.load(root)
        # print("原始数据为{}".format(pre_data[0][0][0]))
        self.data = pre_data
        # batch_ci = 10  # 60000 数据总量为batch_ci和batch_size的积
        # batch_size = 10
        # self.data = torch.zeros((batch_ci * batch_size, image_size, image_size))  # 60000
        #
        # for num in range(batch_ci):
        #     if num % 5000 == 0:
        #         print("Num = %d" % (num + 1))
        #     torch.cuda.empty_cache()
        #
        #     # 初始化 B 数组
        #     B = torch.zeros((iteration, batch_size)).cuda()
        #     BI = torch.zeros((batch_size, iteration, image_size, image_size)).cuda()
        #
        #     # [500, 28, 28]的图像数组
        #     img_batch = torch.zeros((batch_size, image_size, image_size))
        #
        #     # [512]
        #     sum_I1 = torch.zeros(batch_size).cuda()
        #     # [512, 128, 128]
        #     sum_I2 = torch.zeros(img_batch.shape).cuda()
        #     # [512, 128, 128]
        #     sum_ans = torch.zeros(img_batch.shape).cuda()
        #     sum_I2 = sum_I2.permute(1, 2, 0)
        #     sum_ans = sum_ans.permute(1, 2, 0)
        #     # 读取图片
        #     for i in range(batch_size):
        #         x = pre_data[i + batch_size * num]
        #         transform1 = transforms.ToPILImage(mode="L")
        #         y = transform1(np.uint8(x.numpy()))
        #         y = y.resize((image_size, image_size))
        #         transform2 = transforms.ToTensor()
        #         z = transform2(y)
        #         img_batch[i] = z[0]
        #     img_batch = img_batch.cuda()
        #
        #     field = np.random.normal(0, 1, [iteration, image_size, image_size])
        #     field = torch.from_numpy(field).cuda()
        #
        #     for k in range(iteration):
        #         temp = field[k].repeat(batch_size, 1, 1).cuda()
        #         B[k] = torch.mul(temp, img_batch).sum(2).sum(1)
        #         temp = temp.permute(1, 2, 0)
        #         # 桶测量值求和
        #         sum_I1 = sum_I1 + B[k]
        #         # 热光矩阵求和
        #         sum_I2 = sum_I2 + temp
        #         # 桶测量值乘热光矩阵求和
        #         sum_ans = sum_ans + temp * B[k]
        #
        #     sum_I1 = sum_I1 / iteration
        #     sum_I2 = sum_I2 / iteration
        #     sum_ans = sum_ans / iteration
        #     # [512, 128, 128]
        #     ans = sum_ans - sum_I1 * sum_I2
        #     ans = ans.permute(2, 0, 1)
        #     for i in range(batch_size):
        #         self.data[i + batch_size * num] = ans[i]

    #             for i in range(batch_size):
    #                 x = img_batch[i]
    #                 self.data[i + batch_size * num][0] = x

    def __getitem__(self, index):
        img = self.data[index]
        img = img.unsqueeze(0)
        return img

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    data = MY_MNIST_Train(root1=r'C:\Users\chenda\Desktop\Mnist_1on1_2000\train-pt/1.pt',
                               root2=r'C:\Users\chenda\Desktop\Mnist_1on1_2000\train-GI-pt/1.pt')
#     dataloader = DataLoader(data, batch_size=16, shuffle=True,num_workers=0,drop_last=True)
#     for i,(img,label) in  enumerate(dataloader):
#         print(img.shape)
    a, b = data[0]
    print(a.shape)
