import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import mobilenet_rm_filt_pt as mb_pt
import copy

batch_size = 128
num_epochs = 5
learning_rate = 0.001
enable_cuda = True
# enable_cuda = False
load_my_model = True
pruning_method = "chn_prune"
# pruning_method = "mag_prune"
fine_tune = True

# input_size = 32 * 32 * 3 * batch_size
# num_classes = 10

if (torch.cuda.is_available()) and enable_cuda:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

writer = SummaryWriter('runs/mb_cifar10')
#train_dataset = dsets.CIFAR10(root='./data', train=True, transform=transforms.ToTensor(),download=True)
#test_dataset = dsets.CIFAR10(root='./data', train=False, transform=transforms.ToTensor())

train_dataset = dsets.CIFAR10(root='./data', train=True, transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
]), download=True)
test_dataset = dsets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
]))


train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=True)

def load_model(model, path="./mbnv1_pt.pt", print_msg=True):
    try:
        model.load_state_dict(torch.load(path))
        if print_msg:
            print(f"[I] Model loaded from {path}")
    except:
        if print_msg:
            print(f"[E] Model failed to be loaded from {path}")

def model_size(model, count_zeros=True):
    total_params = 0
    nonzero_params = 0
    for tensor in model.parameters():
        t = np.prod(tensor.shape)
        nz = np.sum(tensor.detach().cpu().numpy() != 0.0)

        total_params += t
        nonzero_params += nz
    if not count_zeros:
        return int(nonzero_params)
    else:
        return int(total_params)

class LeNet5(nn.Module):
    def __init__(self, n_classes=10):
        super(LeNet5, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2, bias=False),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, bias=False),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1, bias=False),
            nn.Tanh()
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=120, out_features=84, bias=False),
            nn.Tanh(),
            nn.Linear(in_features=84, out_features=n_classes, bias=False)
        )

        self.mask_dict = None

    def forward(self, x_input):
        # x_input = x.cpu()
        if self.mask_dict:
            self._apply_mask()
        x_input = self.feature_extractor(x_input)
        x_input=torch.flatten(x_input,1)
        logits = self.classifier(x_input)
        probs = F.softmax(logits, dim=1)
        return probs

    def _apply_mask(self):
        for name, param in self.state_dict().items():
            if name in self.mask_dict.keys():
                param.data *= self.mask_dict[name]

# model = LeNet5()
model = mb_pt.MobileNetv1()
model = model.to(torch.device('cuda'))

criterion = nn.CrossEntropyLoss()
criterion_sum = nn.CrossEntropyLoss(reduction='sum')
#optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
iteration = 0

# train_acc = []
# train_loss = []

# test_acc = []
# test_loss = []

def train(epoch, model):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    global iteration
    model.train()
    correct = 0
    total = 0
    for i, (images, labels) in enumerate(train_loader):
        # images = Variable(images.view(-1,1,28,28)).to(torch.device('cuda'))
        images = Variable(images.view(-1,3,32,32)).to(torch.device('cuda'))

        labels = Variable(labels).to(torch.device('cuda'))

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        _, predicted = torch.max(outputs.data,1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
        if (i+1) % 100 == 0:
            print('Epoch: [% d/% d], Step: [% d/% d], Loss: %.4f' % (epoch+1, num_epochs, i+1, len(train_dataset)//batch_size, loss.data.item()))
        iteration += 1
    accuracy = correct.float() / total
    writer.add_scalar("train_accuracy", accuracy, epoch)
    print('Accuracy of the model on the 60000 train images: % f %%' % (100*accuracy))
    # train_acc.append(100 * accuracy)
    # train_loss.append(loss)
    return loss

def test(epoch, mode, value, model):
    correct = 0
    total = 0
    total_loss = 0
    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            # images = Variable(images.view(-1, 1, 28, 28)).to(torch.device('cuda'))
            images = Variable(images.view(-1, 3, 32, 32)).to(torch.device('cuda'))
            labels = Variable(labels).to(torch.device('cuda'))
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()

    accuracy = correct.float() / total
    writer.add_scalar("test_accuracy", accuracy, epoch)
    print(f"mode: {mode}={value}, Test accuracy={100*accuracy.data.item():.4f}")
    # test_acc.append(100 * accuracy)
    return 100*accuracy.data.item()

if load_my_model:
    load_model(model)
    test(0, mode='Non-pruned model', value='True', model=model)
else:
    for epoch in range(num_epochs):
        train(epoch, model)
        test(epoch, mode='test', value='True', model=model)
        torch.save(model.state_dict(),'./checkpoints/saved_model.pt')
    exit(0)

def prune_model_thres(model, threshold):
    mask_dict={}
    for name, param in model.state_dict().items():
        if("weight" in name):
            param[param.abs() < threshold] = 0
            mask_dict[name]=torch.where(torch.abs(param)>0,1,0)
    model.mask_dict = mask_dict

def channel_fraction_pruning(model, fraction=0.2):
    print("In channel fraction pruning function")
    mask_dict={}
    #print(model.state_dict().items())
    #print()
    for name, param in model.state_dict().items():
        #print("The name is " + name)
        #print("The param value is below")
        #print(param)
        #print()
        if ((name == "conv1.weight") or (("layers" in name) and ("conv2" in name))):
            print(name)
            score_list = torch.sum(torch.abs(param),dim=(1,2,3)).to('cpu')
            removed_idx = []
            threshold = np.percentile(np.abs(score_list), fraction*100)
            for i,score in enumerate(score_list):
                if score < threshold:
                    removed_idx.append(i)
                param[removed_idx,:,:,:] = 0
                mask_dict[name]=torch.where(torch.abs(param) > 0,1,0)
            #print(removed_idx)
            #print()
            #print(mask_dict)
            #print()
    model.mask_dict = mask_dict

res1 = []
num_params = []
print("about to prune")
for prune_thres in np.arange(0,1,0.05):
    print(prune_thres)
    if pruning_method=='mag_prune':
        prune_model_thres(model, prune_thres)
    else:
        print("about to fraction prune")
        new_model = copy.deepcopy(model)
        channel_fraction_pruning(new_model, prune_thres)
        print("about to apply mask")
        new_model._apply_mask()
        print("about to remove channel")
        new_model = mb_pt.remove_channel(new_model)
        new_model = new_model.to(torch.device('cuda'))
        #torch.save(model.state_dict(),'mbnv1_pt.pt')
        #load_model(model)

    if fine_tune:
        for fine_tune_epoch in range(num_epochs):
            train(fine_tune_epoch, new_model)

    acc = test(1, "thres", prune_thres, new_model)
    res1.append([prune_thres, acc])
    num_params.append(model_size(new_model, prune_thres==0))
    model_name = "pruned_model_" + str(prune_thres)
    torch.save(new_model.state_dict(), model_name)
res1 = np.array(res1)

#torch.save(new_model.state_dict(),'pruned_model.pt')

plt.figure()
plt.plot(num_params, res1[:,1])
plt.title('{}: Accuracy vs #Params'.format(pruning_method))
plt.xlabel('Number of parameters')
plt.ylabel('Test accuracy')
plt.grid()
# plt.savefig('figs/{}_param_fine_tune_{}.png'.format(pruning_method, fine_tune))
plt.savefig('{}_param_fine_tune_{}.png'.format(pruning_method, fine_tune))
plt.close()

plt.figure()
plt.plot(np.arange(0, 1,0.05), res1[:,1])
plt.title('{}: Accuracy vs Threshold'.format(pruning_method))
plt.xlabel('Pruning Threshold')
plt.ylabel('Test accuracy')
plt.grid()
# plt.savefig('figs/{}_thresh_fine_tune_{}.png'.format(pruning_method, fine_tune))
plt.savefig('{}_thresh_fine_tune_{}.png'.format(pruning_method, fine_tune))
plt.close()