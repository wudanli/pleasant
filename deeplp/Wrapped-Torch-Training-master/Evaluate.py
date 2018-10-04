import torch
from torch import nn
from torch.autograd import Variable

# Evaluate the model
def Evaluate(model, Input, Output, Val_Input, Val_Output, use_gpu = True):
    model.eval()
    criterion = nn.MSELoss()
    if Val_Input.size() and Val_Output.size():
        Input_all = torch.cat((Input, Val_Input), 0)
        Output_all = torch.cat((Output, Val_Output), 0)
    else:
        Input_all = Input
        Output_all = Output
    if use_gpu:
        inputs = Variable(Input_all).cuda()
        outputs = Variable(Output_all).cuda()
    else:       
        inputs = Variable(Input_all)
        outputs = Variable(Output_all)
    predict = model(inputs)
    final_loss = criterion(predict, outputs)
    return predict, final_loss
    