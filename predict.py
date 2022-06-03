import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import torchvision
from torchvision import datasets, models, transforms
import os


import argparse

# まずオブジェクト生成
parser = argparse.ArgumentParser()

# 引数設定
parser.add_argument("--VGG", help="model:VGG16", action="store_true")
parser.add_argument("--AlexNet", help="model:AlexNet", action="store_true")
parser.add_argument("--ResNet", help="model:ResNet", action="store_true")
args = parser.parse_args()

# 参照できる
args.VGG
args.AlexNet
args.ResNet

test_preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_dir = "./train_face/test"
test_dataset = datasets.ImageFolder(test_dir, test_preprocess)

def imshow(images, title=None):
    images = images.numpy().transpose((1, 2, 0))  # (h, w, c)
    # denormalize
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    images = std * images + mean
    images = np.clip(images, 0, 1)
    plt.imshow(images)
    plt.show()

    if title is not None:
        plt.title(title)

if args.VGG == True:
    # モデルの保存
    #weight_file = "D:/ss/ss11/pytorch/logs/epoch001-1.252-0.571.pth"
    weight_file = input("model path:")
    model = models.vgg16(pretrained=False)
    print("load models")
    model.classifier = nn.Sequential(
        nn.Linear(25088, 4096),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(4096, 4096),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(4096, 4)
    )

    model.load_state_dict(torch.load(weight_file,
                                     map_location=lambda storage,
                                     loc: storage))


    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=32,
                                              shuffle=False)

    images, _ = iter(test_loader).next()
    with torch.no_grad():
        print(images.size())
    images, classes = next(iter(test_loader))
    print(images.size(), classes.size())
    grid = torchvision.utils.make_grid(images[:16], nrow=4)
    imshow(grid)

    # ネットワークに通して予測する
    outputs = model(images)
    _, predicted = torch.max(outputs.data, 1)

    # 予測結果がラベルで表示される
    print(predicted.numpy())

elif args.AlexNet == True:
    print("AlexNet")
    # モデルの保存
    #weight_file = "D:/ss/ss11/pytorch/logs/epoch001-1.252-0.571.pth"
    weight_file = input("model path:")
    model = models.alexnet(pretrained=False)
    print("load models")
    # 全層のパラメータを固定
    for param in model.parameters():
        param.requires_grad = False

    # 最終層を4クラス用に変更.
    # 変更することで、最終層はbackward時に重みが更新される
    num_ftrs = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_ftrs, 4)

    model.load_state_dict(torch.load(weight_file,
                                     map_location=lambda storage,
                                     loc: storage))


    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=32,
                                              shuffle=False)

    images, _ = iter(test_loader).next()
    with torch.no_grad():
        print(images.size())

    images, classes = next(iter(test_loader))
    print(images.size(), classes.size())
    grid = torchvision.utils.make_grid(images[:16], nrow=4)
    imshow(grid)

    # ネットワークに通して予測する
    outputs = model(images)
    _, predicted = torch.max(outputs.data, 1)

    # 予測結果がラベルで表示される
    print(predicted.numpy())

elif args.ResNet == True:
    print("ResNet")
    # モデルの保存
    weight_file = input("model path:")
    model = models.resnet18(pretrained=False)
    # 分類を4クラスにする
    num_features = model.fc.in_features
    print(num_features)  # 512

    # fc層を置き換える
    model.fc = nn.Linear(num_features, 4)

    print("load models")
    model.load_state_dict(torch.load(weight_file,
                                     map_location=lambda storage,
                                                         loc: storage))

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=32,
                                              shuffle=False)

    images, _ = iter(test_loader).next()
    with torch.no_grad():
        print(images.size())

    images, classes = next(iter(test_loader))
    print(images.size(), classes.size())
    grid = torchvision.utils.make_grid(images[:16], nrow=4)
    imshow(grid)

    # ネットワークに通して予測する
    outputs = model(images)
    _, predicted = torch.max(outputs.data, 1)

    # 予測結果がラベルで表示される
    print(predicted.numpy())
