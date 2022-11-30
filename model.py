import torch
import torch.nn.functional as F
from torch import nn
from torchsummary import summary
import torchvision.models as models

# CNN model
class CNN(nn.Module):
    def __init__(self, hidden_size, output_size, n_layers=1, dropout_p=0.001):
        super(CNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.dropout = nn.Dropout(self.dropout_p)

        self.conv1 = nn.Conv2d(12,32,(1,100),(1,3))
        self.conv2 = nn.Conv2d(32,32,(1,25),(1,3))
        self.conv3 = nn.Conv2d(32,16,(1,5),(1,3))

        self.bn = nn.BatchNorm2d(16)

        self.fc1 = nn.Linear(16*21, self.hidden_size)
        #self.fc2 = nn.Linear(self.hidden_size, 128)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, encoder_outputs):
        cnn_out = F.max_pool2d(F.relu(self.conv1(encoder_outputs)),(1,2),(1,2))
        cnn_out = F.max_pool2d(F.relu(self.conv2(cnn_out)),(1,2),(1,2))
        cnn_out = F.max_pool2d(F.relu(self.conv3(cnn_out)),(1,2),(1,2))
        
        cnn_out = cnn_out.view(-1,16*21)
        output1 = F.relu(self.fc1(cnn_out))
        #output2 = F.relu(self.fc2(output1))
        output = self.out(output1)
        
        return output

def model():
    model = CNN(256,7)
    return model