import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from tqdm import tqdm

from deepfool import deepfool
from train_model import model
from PIL import Image

batch_size = 10    # 批处理尺寸
adver_nums = 1000
# 定义GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = model.ResNet18()
adver_example_DeepFool = torch.zeros((batch_size, 3, 32, 32)).to(device)
adver_target = torch.zeros(batch_size).to(device)
clean_example = torch.zeros((batch_size, 3, 32, 32)).to(device)
clean_target = torch.zeros(batch_size).to(device)

# 加载并预处理数据集
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),   # 随机裁剪
    transforms.RandomHorizontalFlip(),      # 图像翻转
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# 下载数据集
trainset = torchvision.datasets.CIFAR10(root='../../data', train=True, download=True, transform=transform_train)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='../../data', train=False, download=True, transform=transform_test)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)


for i, (images, targets) in enumerate(testloader):
    images, targets = images.to(device), targets.to(device)
    if i >= adver_nums/batch_size:
        break
    if i == 0:
        clean_example = images
    else:
        clean_example = torch.cat((clean_example, images), dim=0)

    cur_adver_example = torch.zeros_like(images).to(device)

    for j in range(batch_size):
        r_tot, loop_i, label, k_i, pert_image = deepfool(images[j], net)
        cur_adver_example[j] = pert_image

    pred = net(cur_adver_example).max(1)[1]

    if i == 0:
        adver_example_DeepFool = cur_adver_example
        clean_target = targets
        adver_target = pred

    else:
        adver_example_DeepFool = torch.cat((adver_example_DeepFool, cur_adver_example), dim=0)
        clean_target = torch.cat((clean_target, targets), dim=0)
        adver_target = torch.cat((adver_target, pred), dim=0)


print(adver_example_DeepFool.shape)
print(adver_target.shape)
print(clean_example.shape)
print(clean_target.shape)

# 使用对抗样本攻击模型
def adver_attack_model(model, adver_example, target, name):

    """print the correct number of the model and accuracy of the model.
       :param model: model to attack.
       :param adver_example: the adversarial example we use.
       :param target: the targets of examples.
       :param name: the name of model.
       :return:None
    """
    adver_dataset = TensorDataset(adver_example, target)

    loader = DataLoader(dataset=adver_dataset, batch_size=batch_size)
    correct_num = torch.tensor(0).to(device)
    for j, (images, targets) in tqdm(enumerate(loader)):
        images, targets = images.to(device), targets.to(device)
        pred = model.forward(images).max(1)[1]
        num = torch.sum(pred == targets)
        correct_num = correct_num + num
        print(correct_num)
    print(correct_num)
    print('\n{} correct rate is {}'.format(name, correct_num/adver_nums))

adver_attack_model(net, adver_example=adver_example_DeepFool, target=clean_target, name="ResNet18")


# 可视化展示
def plot_clean_and_adver(adver_example, adver_target, clean_example, clean_target):
    n_cols = 5
    n_rows = 5
    cnt = 1
    cnt1 = 1
    plt.figure(figsize=(n_cols*4, n_rows*2))
    for i in range(n_cols):
        for j in range(n_cols):
            plt.subplot(n_cols, n_rows*2, cnt1)
            plt.xticks([])
            plt.yticks([])
            plt.title("{} -> {}".format(clean_target[cnt], adver_target[cnt]))
            plt.imshow((clean_example[cnt].permute(1, 2, 0).to("cpu").detach().numpy()*255).astype(np.uint8))
            plt.subplot(n_cols, n_rows*2, cnt1+1)
            plt.xticks([])
            plt.yticks([])
            plt.imshow((adver_example[cnt].permute(1, 2, 0).to("cpu").detach().numpy()*255).astype(np.uint8))
            cnt = cnt + 1
            cnt1 = cnt1 + 2

    plt.show()
plot_clean_and_adver(adver_example=adver_example_DeepFool, adver_target=adver_target, clean_example=clean_example, clean_target=clean_target)
