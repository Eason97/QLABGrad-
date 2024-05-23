import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torch.utils.data as data
import torchvision.transforms as transforms
import QLABplr
import torch.nn.functional as F
trainloader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=True, download=True,transform=transforms.Compose([transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))])),batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(datasets.MNIST('./data', train=False, transform=transforms.Compose([
    transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])),batch_size=1, shuffle=True)
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self._input_dim = input_dim
        self.lin1 = nn.Linear(input_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)
        self.lin3 = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        x = x.view(-1, self._input_dim)
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = self.lin3(x)
        return x
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = MLP(28 * 28, 1000, 10)
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimal_lr=1e-5
iteration=0
for epoch in range(1):
    model.eval()
    loss_epoch=0
    for batch_idx, (data, target) in enumerate(trainloader):
        if batch_idx<50:
            data, target = data.cuda(), target.cuda()
            g_0 = criterion(model(data), target)
            optimal_lr = QLABplr.FindPLR(3,optimal_lr,model,criterion,data,target)
            print('optimal_lr',batch_idx,optimal_lr)
        else:
            break


