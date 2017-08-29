import torch
import torch.nn as nn
from torchvision import models
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from sklearn.preprocessing import MultiLabelBinarizer

#used for all models
class NetWithSigmoid(nn.Module):
    def __init__(self, modules_lst):
        super(NetWithSigmoid, self).__init__()
        #want correct ordering of layers, so we're using modules_lst
        #for key, module in modules_dict.items():
        #    self.add_module(key, module)
        for child in modules_lst:
            self.add_module(child[0], child[1])
            
    def forward(self, x):
        list = []
        for name, module in self._modules.items():
            if name == 'fc':
                x = x.view(x.size(0), -1) # Flatten layer
            x = module(x)
        x = F.sigmoid(x)
        return x


class ResNet(nn.Module):

    def __init__(self, modules_lst):
        #creating resnet with an adjusted final connected layer to output to 17 neurons
        #but without final sigmoid layer
        model_without_sigmoid = models.resnet34(pretrained=True)
        model_without_sigmoid._modules['fc'] = nn.Linear(512, 17)
        #to preserve the order of the layers
        #modules_dict = {child[0]:child[1] for child in list(model_without_sigmoid.named_children())}
        modules_lst = list(model_without_sigmoid.named_children())
        #model_with_sig = NetWithSigmoid(modules_dict)
        model_with_sig = NetWithSigmoid(modules_lst).cuda()

        #setting learning rates slower for everything except the last layer
        ignored_params = list(map(id, model_with_sig.fc.parameters()))
        base_params = filter(lambda p: id(p) not in ignored_params, model_with_sig.parameters())

    
    def forward(self, x):
        return model_without_sigmoid.forward(x)

