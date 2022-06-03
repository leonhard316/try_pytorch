import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
#from torch.autograd import Variable
import torchvision
from torchvision import datasets, models, transforms
import os
import time
import argparse
import copy

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

# 訓練とバリデーションの関数の定義
def train(model, criterion, optimizer, train_loader):
    model.train()
    running_loss = 0
    for batch_idx, (images, labels) in enumerate(train_loader):
        # if use_gpu:
        # images = Variable(images.cuda())
        # labels = Variable(labels.cuda())
        # else:
        # images = Variable(images)
        # labels = Variable(labels)

        optimizer.zero_grad()
        outputs = model(images)

        loss = criterion(outputs, labels)
        running_loss += loss.data.item()

        loss.backward()
        optimizer.step()

    train_loss = running_loss / len(train_loader)
    return train_loss

def valid(model, criterion, valid_loader):
    model.eval()
    running_loss = 0
    correct = 0
    total = 0
    for batch_idx, (images, labels) in enumerate(valid_loader):
        # if use_gpu:
        # images = Variable(images.cuda(), volatile=True)
        # labels = Variable(labels.cuda(), volatile=True)
        # else:
        with torch.no_grad():
            # images = Variable(images, volatile=True)
            # labels = Variable(labels, volatile=True)

            outputs = model(images)

            loss = criterion(outputs, labels)
            running_loss += loss.data.item()

            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels.data).sum()
            total += labels.size(0)

    val_loss = running_loss / len(valid_loader)
    val_acc = correct / total

    return val_loss, val_acc


# 前処理　データ拡張含む
train_preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
}

# データセットの作成
train_dir = "./train_face/train"
valid_dir = "./train_face/valid"

train_dataset = datasets.ImageFolder(train_dir, train_preprocess)
valid_dataset = datasets.ImageFolder(valid_dir, test_preprocess)

classes = train_dataset.classes
print(train_dataset.classes)
print(valid_dataset.classes)

# データをミニバッチ単位で取得
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=8,
                                           shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                           batch_size=8,
                                           shuffle=False)

images, classes = next(iter(train_loader))
print(images.size(), classes.size())

if args.VGG == True:
    print("VGG16")

    # モデルのダウンロード
    vgg16 = models.vgg16(pretrained=True)
    print(vgg16)

    # 全層のパラメータを固定
    for param in vgg16.parameters():
        param.requires_grad = False

    # classifierの出力層を1000→4に変更
    vgg16.classifier = nn.Sequential(
        nn.Linear(25088, 4096),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(4096, 4096),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(4096, 4)
    )

    # モデルの訓練
    # loss関数の定義
    criterion = nn.CrossEntropyLoss()
    # 最適化手法のパラメータ設定
    optimizer = optim.SGD(vgg16.classifier.parameters(), lr=0.001, momentum=0.9)

    num_epochs = 200
    log_dir = './logs/VGG16'
    # ディレクトリの作成
    if os.path.exists(log_dir) == False:
        os.mkdir(log_dir)

    best_acc = 0
    loss_list = []
    val_loss_list = []
    val_acc_list = []

    for epoch in range(num_epochs):
        loss = train(vgg16, criterion, optimizer, train_loader)
        val_loss, val_acc = valid(vgg16, criterion, valid_loader)

        print('epoch %d, loss: %.4f val_loss: %.4f val_acc: %.4f'
              % (epoch, loss, val_loss, val_acc))

        if val_acc > best_acc:
            print('val_acc improved from %.5f to %.5f!' % (best_acc, val_acc))
            best_acc = val_acc
            model_file = 'epoch%03d-%.3f-%.3f.pth' % (epoch, val_loss, val_acc)
            torch.save(vgg16.state_dict(), os.path.join(log_dir, model_file))

        # logging
        loss_list.append(loss)
        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc)

elif args.AlexNet == True:
    print("AlexNet")
    # モデルのダウンロード
    AlexNet = models.alexnet(pretrained=True)
    print(AlexNet)

    # 全層のパラメータを固定
    for param in AlexNet.parameters():
        param.requires_grad = False

    # 最終層を4クラス用に変更.
    # 変更することで、最終層はbackward時に重みが更新される
    num_ftrs = AlexNet.classifier[6].in_features
    AlexNet.classifier[6] = nn.Linear(num_ftrs, 4)

    # 前処理
    train_preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # データセットの作成
    train_dir = "./train_face/train"
    valid_dir = "./train_face/valid"

    train_dataset = datasets.ImageFolder(train_dir, train_preprocess)
    valid_dataset = datasets.ImageFolder(valid_dir, test_preprocess)

    classes = train_dataset.classes
    print(train_dataset.classes)
    print(valid_dataset.classes)

    # データをミニバッチ単位で取得
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=8,
                                               shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                               batch_size=8,
                                               shuffle=False)

    images, classes = next(iter(train_loader))
    print(images.size(), classes.size())

    # モデルの訓練
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(AlexNet.classifier.parameters(), lr=0.001, momentum=0.9)

    num_epochs = 200
    log_dir = './logs/AlexNet'
    if os.path.exists(log_dir) == False:
        os.mkdir(log_dir)

    best_acc = 0
    loss_list = []
    val_loss_list = []
    val_acc_list = []
    for epoch in range(num_epochs):
        loss = train(AlexNet, criterion, optimizer, train_loader)
        val_loss, val_acc = valid(AlexNet, criterion, valid_loader)

        print('epoch %d, loss: %.4f val_loss: %.4f val_acc: %.4f'
              % (epoch, loss, val_loss, val_acc))

        if val_acc > best_acc:
            print('val_acc improved from %.5f to %.5f!' % (best_acc, val_acc))
            best_acc = val_acc
            model_file = 'epoch%03d-%.3f-%.3f.pth' % (epoch, val_loss, val_acc)
            torch.save(AlexNet.state_dict(), os.path.join(log_dir, model_file))

        # logging
        loss_list.append(loss)
        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc)
        # Early Stopping チェック

elif args.ResNet == True:
    print("ResNet")

    # モデルのダウンロード
    ResNet = models.resnet18(pretrained=True)
    print(ResNet)

    # すべてのパラメータを固定
    for param in ResNet.parameters():
        param.requires_grad = False

    # 分類を4クラスにする
    num_features = ResNet.fc.in_features
    print(num_features)  # 512

    # fc層を置き換える
    ResNet.fc = nn.Linear(num_features, 4)
    print(ResNet)

    train_size = int(len(train_dataset))
    validation_size = int(len(valid_dataset))
    data_size = {"train": train_size, "val": validation_size}

    dataloaders = {"train": train_loader, "val": valid_loader}
    log_dir = './logs/AlexNet'
    if os.path.exists(log_dir) == False:
        os.mkdir(log_dir)

    def train_model(model, criterion, optimizer, scheduler, num_epochs=200):
        since = time.time()

        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0

        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

            # 各エポックで訓練+バリデーションを実行
            for phase in ['train', 'val']:
                if phase == 'train':
                    scheduler.step()
                    model.train(True)  # training mode
                else:
                    model.train(False)  # evaluate mode

                running_loss = 0.0
                running_corrects = 0

                for data in dataloaders[phase]:
                    inputs, labels = data
                    optimizer.zero_grad()

                    # forward
                    outputs = model(inputs)
                    _, preds = torch.max(outputs.data, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                # サンプル数で割って平均を求める
                epoch_loss = running_loss / data_size[phase]
                epoch_acc = running_corrects / data_size[phase]

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

                # deep copy the model
                # 精度が改善したらモデルを保存する
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

            print()

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val acc: {:.4f}'.format(best_acc))

        # load best model weights
        model.load_state_dict(best_model_wts)
        return model

    # モデルの訓練
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(ResNet.parameters(), lr=0.001, momentum=0.9)
    # 7エポックごとに学習率を0.1倍する
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    ResNet = train_model(ResNet, criterion, optimizer, exp_lr_scheduler, num_epochs=200)
    torch.save(ResNet.state_dict(), './logs/ResNet/model_ft.pkl')
