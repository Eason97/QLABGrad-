import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from torchvision.datasets.mnist import MNIST
import torchvision.transforms as transforms
from collections import OrderedDict
import qlab
from tensorboardX import SummaryWriter

data_train = MNIST('data/mnist',
                   download=True,
                   transform=transforms.Compose([
                       transforms.Resize((32, 32)),
                       transforms.ToTensor()]))
data_test = MNIST('data/mnist',
                  train=False,
                  download=True,
                  transform=transforms.Compose([
                      transforms.Resize((32, 32)),
                      transforms.ToTensor()]))
data_train_loader = DataLoader(data_train, batch_size=64, shuffle=True, num_workers=8)
data_test_loader = DataLoader(data_test, batch_size=64, num_workers=8)

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()

        in_channels = 1
        self.model = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(1024, 1000)), 
            ('relu1', nn.ReLU()), 
            ('f2', nn.Linear(1000, 1000)), 
            ('relu2', nn.ReLU()), 
            ('f3', nn.Linear(1000, 10)) 
        ]))

    def forward(self, img):
        output = img.view(img.size(0), -1)
        output = self.model(output) 
        return output


device = torch.device("cuda")
cudnn.benchmark = True
torch.manual_seed(1)
writer = SummaryWriter()

def test(net, criterion, _data_loader, _dataset):
    global device
    with torch.no_grad():
        net.eval()
        total_correct = 0
        avg_loss = 0.0
        for i, (images, labels) in enumerate(_data_loader):
            images, labels = images.to(device), labels.to(device)
            output = net(images)
            avg_loss += criterion(output, labels).sum()
            pred = output.detach().max(1)[1]
            total_correct += pred.eq(labels.view_as(pred)).sum()
        avg_loss /= len(_dataset)
        avg_loss_value = avg_loss.detach().cpu().item()
        acc = float(total_correct) / len(_dataset)
        return [avg_loss_value, acc]



net = MLP().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01)

iteration = 0
prev_lr = 1e-1
for epoch in range(0, 100):
    print('Starting epoch %d / %d' % (epoch + 1, 50))
    net.train()
    for i, (images, labels) in enumerate(data_train_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        output = net(images)
        loss = criterion(output, labels)
        loss.backward()
        optimal_lr= qlab.qlab_delta(optimizer, net, criterion, images, labels,prev_lr)
        optimizer.step()
        prev_lr = optimal_lr
        writer.add_scalars('QLAB LR', {'LR': optimal_lr}, iteration)
        iteration += 1

    [train_avg_loss, train_acc] = test(net, criterion, data_train_loader, data_train)
    writer.add_scalars('Train Avg Loss', {'Train Avg Loss': train_avg_loss}, epoch)
    writer.add_scalars('Train Acc', {'Train Acc': train_acc}, epoch)

    [test_avg_loss, test_acc] = test(net, criterion, data_test_loader, data_test)
    writer.add_scalars('Test Avg Loss', {'Test Avg Loss': test_avg_loss}, epoch)
    writer.add_scalars('Test Acc', {'Test Acc': test_acc}, epoch)

