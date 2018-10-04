import torch
from torch import nn
import torch.nn.functional as F

# Full_Connected_Network
class Full_Connected(nn.Module):
    def __init__(self, D_Inputs, D_Outputs, Hidden_Layer_Structure, Activation = 'relu'):
        super(Full_Connected, self).__init__()
        self.Activation = Activation
        self.Hidden_Layer_Structure = Hidden_Layer_Structure
        for i in range(len(Hidden_Layer_Structure) + 1):
            if i == 0:
                exec('self.fc' + str(i + 1) + ' = nn.Linear(' +
                     str(D_Inputs) + ', ' + str(Hidden_Layer_Structure[i]) + ')')
            elif i == len(Hidden_Layer_Structure):
                exec('self.fc' + str(i + 1) + ' = nn.Linear(' +
                     str(Hidden_Layer_Structure[i - 1]) + ', ' + str(D_Outputs) + ')')
            else:
                exec('self.fc' + str(i + 1) + ' = nn.Linear(' +
                     str(Hidden_Layer_Structure[i - 1]) + ', ' + str(Hidden_Layer_Structure[i]) + ')')
    def forward(self, x):
        left = ''
        right = ''
        for i in range(len(self.Hidden_Layer_Structure) + 1):
            if i == len(self.Hidden_Layer_Structure):
                left = 'self.fc' + str(i + 1) +'(' + left
                right = right + ')'
            else:
                left = 'F.' + self.Activation + '(self.fc' + str(i + 1) +'(' + left
                right = right + '))'
        x = eval(left + 'x' + right)
        return x

if __name__ == '__main__':
    # Test of Full_Connected_Network
    Hidden_Layer_Structure = [298, 299, 300, 301]
    Activation = 'relu'
    D_Inputs, D_Outputs = 26, 33
    model = Full_Connected(D_Inputs, D_Outputs, Hidden_Layer_Structure, Activation = Activation).double()
    if torch.cuda.is_available():
        model = model.cuda()