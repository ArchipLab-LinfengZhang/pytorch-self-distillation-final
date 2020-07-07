import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
from resnet import *
import torch.nn.functional as F
from autoaugment import CIFAR10Policy
from cutout import Cutout
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='Self-Distillation CIFAR Training')
parser.add_argument('--model', default="resnet18", type=str, help="resnet18|resnet34|resnet50|resnet101|resnet152|"
                                                                   "wideresnet50|wideresnet101|resnext50|resnext101")
parser.add_argument('--dataset', default="cifar100", type=str, help="cifar100|cifar10")
parser.add_argument('--epoch', default=250, type=int, help="training epochs")
parser.add_argument('--loss_coefficient', default=0.3, type=float)
parser.add_argument('--feature_loss_coefficient', default=0.03, type=float)
parser.add_argument('--dataset_path', default="data", type=str)
parser.add_argument('--autoaugment', default=True, type=bool)
parser.add_argument('--temperature', default=3.0, type=float)
parser.add_argument('--batchsize', default=128, type=int)
parser.add_argument('--init_lr', default=0.1, type=float)
args = parser.parse_args()
print(args)

def CrossEntropy(outputs, targets):
    log_softmax_outputs = F.log_softmax(outputs/args.temperature, dim=1)
    softmax_targets = F.softmax(targets/args.temperature, dim=1)
    return -(log_softmax_outputs * softmax_targets).sum(dim=1).mean()


if args.autoaugment:
    transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4, fill=128),
                             transforms.RandomHorizontalFlip(), CIFAR10Policy(), transforms.ToTensor(),
                             Cutout(n_holes=1, length=16),
                             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
else:
    transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4, fill=128),
                                          transforms.RandomHorizontalFlip(), transforms.ToTensor(),
                                          transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                               (0.2023, 0.1994, 0.2010))])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
if args.dataset == "cifar100":
    trainset = torchvision.datasets.CIFAR100(
        root=args.dataset_path,
        train=True,
        download=True,
        transform=transform_train
    )
    testset = torchvision.datasets.CIFAR100(
        root=args.dataset_path,
        train=False,
        download=True,
        transform=transform_test
    )
elif args.dataset == "cifar10":
    trainset = torchvision.datasets.CIFAR10(
        root=args.dataset_path,
        train=True,
        download=True,
        transform=transform_train
    )
    testset = torchvision.datasets.CIFAR10(
        root=args.dataset_path,
        train=False,
        download=True,
        transform=transform_test
    )
trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=args.batchsize,
    shuffle=True,
    num_workers=4
)
testloader = torch.utils.data.DataLoader(
    testset,
    batch_size=args.batchsize,
    shuffle=False,
    num_workers=4
)

if args.model == "resnet18":
    net = resnet18()
if args.model == "resnet34":
    net = resnet34()
if args.model == "resnet50":
    net = resnet50()
if args.model == "resnet101":
    net = resnet101()
if args.model == "resnet152":
    net = resnet152()
if args.model == "wideresnet50":
    net = wide_resnet50_2()
if args.model == "wideresnet101":
    net = wide_resnet101_2()
if args.model == "resnext50_32x4d":
    net = resnet18()
if args.model == "resnext101_32x8d":
    net = resnext101_32x8d()

net.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.init_lr, weight_decay=5e-4, momentum=0.9)
init = False

if __name__ == "__main__":
    best_acc = 0
    for epoch in range(args.epoch):
        correct = [0 for _ in range(5)]
        predicted = [0 for _ in range(5)]
        if epoch in [args.epoch // 3, args.epoch * 2 // 3, args.epoch - 10]:
            for param_group in optimizer.param_groups:
                param_group['lr'] /= 10
        net.train()
        sum_loss, total = 0.0, 0.0
        for i, data in enumerate(trainloader, 0):
            length = len(trainloader)
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs, outputs_feature = net(inputs)
            ensemble = sum(outputs[:-1])/len(outputs)
            ensemble.detach_()

            if init is False:
                #   init the adaptation layers.
                #   we add feature adaptation layers here to soften the influence from feature distillation loss
                #   the feature distillation in our conference version :  | f1-f2 | ^ 2
                #   the feature distillation in the final version : |Fully Connected Layer(f1) - f2 | ^ 2
                layer_list = []
                teacher_feature_size = outputs_feature[0].size(1)
                for index in range(1, len(outputs_feature)):
                    student_feature_size = outputs_feature[index].size(1)
                    layer_list.append(nn.Linear(student_feature_size, teacher_feature_size))
                net.adaptation_layers = nn.ModuleList(layer_list)
                net.adaptation_layers.cuda()
                optimizer = optim.SGD(net.parameters(), lr=args.init_lr, weight_decay=5e-4, momentum=0.9)
                #   define the optimizer here again so it will optimize the net.adaptation_layers
                init = True

            #   compute loss
            loss = torch.FloatTensor([0.]).to(device)

            #   for deepest classifier
            loss += criterion(outputs[0], labels)

            teacher_output = outputs[0].detach()
            teacher_feature = outputs_feature[0].detach()

            #   for shallow classifiers
            for index in range(1, len(outputs)):
                #   logits distillation
                loss += CrossEntropy(outputs[index], teacher_output) * args.loss_coefficient
                loss += criterion(outputs[index], labels) * (1 - args.loss_coefficient)
                #   feature distillation
                if index != 1:
                    loss += torch.dist(net.adaptation_layers[index-1](outputs_feature[index]), teacher_feature) * \
                            args.feature_loss_coefficient
                    #   the feature distillation loss will not be applied to the shallowest classifier

            sum_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(labels.size(0))
            outputs.append(ensemble)

            for classifier_index in range(len(outputs)):
                _, predicted[classifier_index] = torch.max(outputs[classifier_index].data, 1)
                correct[classifier_index] += float(predicted[classifier_index].eq(labels.data).cpu().sum())
            print('[epoch:%d, iter:%d] Loss: %.03f | Acc: 4/4: %.2f%% 3/4: %.2f%% 2/4: %.2f%%  1/4: %.2f%%'
                  ' Ensemble: %.2f%%' % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1),
                                          100 * correct[0] / total, 100 * correct[1] / total,
                                          100 * correct[2] / total, 100 * correct[3] / total,
                                          100 * correct[4] / total))

        print("Waiting Test!")
        with torch.no_grad():
            correct = [0 for _ in range(5)]
            predicted = [0 for _ in range(5)]
            total = 0.0
            for data in testloader:
                net.eval()
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs, outputs_feature = net(images)
                ensemble = sum(outputs) / len(outputs)
                outputs.append(ensemble)
                for classifier_index in range(len(outputs)):
                    _, predicted[classifier_index] = torch.max(outputs[classifier_index].data, 1)
                    correct[classifier_index] += float(predicted[classifier_index].eq(labels.data).cpu().sum())
                total += float(labels.size(0))

            print('Test Set AccuracyAcc: 4/4: %.4f%% 3/4: %.4f%% 2/4: %.4f%%  1/4: %.4f%%'
                  ' Ensemble: %.4f%%' % (100 * correct[0] / total, 100 * correct[1] / total,
                                         100 * correct[2] / total, 100 * correct[3] / total,
                                         100 * correct[4] / total))
            if correct[4] / total > best_acc:
                best_acc = correct[4]/total
                print("Best Accuracy Updated: ", best_acc * 100)
                torch.save(net.state_dict(), "./checkpoints/"+str(args.model)+".pth")

    print("Training Finished, TotalEPOCH=%d, Best Accuracy=%.3f" % (args.epoch, best_acc))
