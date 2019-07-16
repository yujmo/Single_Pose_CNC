import torch
import torch.nn as nn
import torch.optim as optim
from data import pose_set
from torch.optim.lr_scheduler import ReduceLROnPlateau
from visdom import Visdom

batch_size = 128
learning_rate = 2e-1
epochs = 35

# Train_set
train_pose = pose_set('E:\\l\\train.csv')
train_loader = torch.utils.data.DataLoader(dataset=train_pose,
                                           batch_size=batch_size,
                                           shuffle=True)
# Test_set
test_pose = pose_set('E:\\l\\test.csv')
test_loader = torch.utils.data.DataLoader(dataset=test_pose,
                                          batch_size=batch_size,
                                          shuffle=True)

""" train_pose, val_pose = torch.utils.data.random_split(train_pose, [14000, 2148])

train_loader = torch.utils.data.DataLoader(
    train_pose,
    batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(
    val_pose,
    batch_size=batch_size, shuffle=True) """


class MLP(nn.Module):

    def __init__(self):
        super(MLP, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(18, 384),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(384, 384),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(384, 384),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(384, 3),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        x = self.model(x)
        return x


device = torch.device('cuda:0')
net = MLP().double().to(device)
#optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=0.3)
#optimizer = optim.SGD(net.parameters(), lr=learning_rate, weight_decay=0.01)
optimizer = optim.SGD(net.parameters(), lr=learning_rate,
                      momentum=0.50)
scheduler = ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=3, verbose=True)
criteon = nn.CrossEntropyLoss().to(device)

viz = Visdom()

viz.line([0.], [0.], win='train_loss', opts=dict(title='train loss'))

global_step = 0

for epoch in range(epochs):

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        logits = net(data)

        optimizer.zero_grad()
        loss = criteon(logits, target)
        loss.backward()
        optimizer.step()

        global_step += 1
        viz.line([loss.item()], [global_step],
                 win='train_loss', update='append')

        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))

    test_loss = 0
    correct = 0
    lac_c = 0
    loc_c = 0
    nc_c = 0
    lac = 0
    nc = 0
    loc = 0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        logits = net(data)
        test_loss += criteon(logits, target).item()

        pred = logits.data.max(1)[1]
        pred_shape = pred.size()
        correct += pred.eq(target.data).sum()

        for i in range(pred_shape[0]):
            if target.data[i] == 0:
                nc_c += pred.eq(target.data)[i].item()
                nc += 1
            elif target.data[i] == 2:
                loc_c += pred.eq(target.data)[i].item()
                loc += 1
            else:
                lac_c += pred.eq(target.data)[i].item()
                lac += 1
        
        if epoch == 30:
            print('+++++++++++++++++')
            print(pred)
            print(target.data)

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    scheduler.step(test_loss)
    print('Accuracy of lat Cross : %2d %%' % (
        100 * lac_c / lac))
    print(lac_c)
    print('Accuracy of Not-Cross : %2d %%' % (
        100 * nc_c / nc))
    print(nc_c)
    print('Accuracy of lon Cross : %2d %%' % (
        100 * loc_c / loc))
    print(loc_c)

torch.save(net.state_dict(), 'net.pkl')
    



test_loss = 0
correct = 0
for data, target in test_loader:
    data, target = data.to(device), target.to(device)
    logits = net(data)
    test_loss += criteon(logits, target).item()

    pred = logits.data.max(1)[1]
    correct += pred.eq(target.data).sum()

test_loss /= len(test_loader.dataset)
print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))