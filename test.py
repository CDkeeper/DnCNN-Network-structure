import time
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from dataset import MY_MNIST_Test
from torchvision.utils import save_image
batch_size = 100
toPIL = transforms.ToPILImage()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# field = np.random.normal(0, 1, [iteration, image_size, image_size])
# field = torch.from_numpy(field).cuda()

test_dataset = MY_MNIST_Test(root=r'C:\Users\chenda\Desktop\Mnist_1on1_dncnn_2000_100_128\test-GI-pt/1.pt')

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
model = torch.load("./model_weight/module_100.pth").to(device)
# print(model)
print("模型加载成功！")
start_time = time.time()
for batch_idx, data in enumerate(test_loader):
    model.eval()
    # print(data.shape)
    data = data.to(device)
    with torch.no_grad():
        data = data.to(torch.float32)
        output = model(data)
        output = output.cpu()
    for i in range(batch_size):  # 读取每一批图片，并分离
        img = output[i]
        save_image(img, r'C:\Users\chenda\Desktop\Mnist_1on1_dncnn_2000_100_128\test-result/' + str(i + 1) + '.jpg'.format(
            (batch_idx * batch_size + i + 1)), nrow=1,
                   scale_each=True)
        # pic = toPIL(img)
        # pic.save(r"C:\Users\chenda\Desktop\Mnist_1on1_dncnn_2000_100_128\test-result/{}.jpg".format((batch_idx * batch_size + i + 1)))

end_time = time.time()
print("预测{}张图片累计耗时{}s".format(len(test_dataset), end_time - start_time))

