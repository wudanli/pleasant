from torch import nn, optim
from Data_Reading import Data_Reading
from torch.autograd import Variable
from Networks import Full_Connected

# SGD_Method
def SGD_method(model, Input, Output, Val_input, Val_output, num_epochs = 100, 
               learning_rate = 1e-4, use_gpu = True, 
               Loss_Function = 'MSELoss', exp_step = 20):
    criterion = eval('nn.' + Loss_Function + '()')
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=0)
    if Val_input.size() and Val_output.size():
        for epoch in range(num_epochs):
            if use_gpu:
                inputs = Variable(Input).cuda()
                outputs = Variable(Output).cuda()
                val_inputs = Variable(Val_input).cuda()
                val_outputs = Variable(Val_output).cuda()
            else:       
                inputs = Variable(Input)
                outputs = Variable(Output)
                val_inputs = Variable(Val_input)
                val_outputs = Variable(Val_output)
            #forward
            out = model(inputs)
            val_out = model(val_inputs)
            loss = criterion(out,outputs)
            val_loss = criterion(val_out,val_outputs)
            #backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (epoch + 1) % exp_step == 0:
                print('Epoch[{}/{}], loss: {:.12f}, val_loss:{:.12f}'
                      .format(epoch + 1, num_epochs, loss.data[0], val_loss.data[0]))
        return model, loss, val_loss
    else:
        for epoch in range(num_epochs):
            if use_gpu:
                inputs = Variable(Input).cuda()
                outputs = Variable(Output).cuda()
            else:       
                inputs = Variable(Input)
                outputs = Variable(Output)
            # Forward
            out = model(inputs)
            loss = criterion(out, outputs)
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (epoch + 1) % exp_step == 0:
                print('Epoch[{}/{}], loss: {:.12f}'.format(epoch + 1, num_epochs, loss.data[0]))
        return model, loss
# LBFGS Method
def LBFGS_method(model, Input, Output, Val_input, Val_output, num_epochs = 100, 
                 use_gpu = True, Loss_Function = 'MSELoss', exp_step = 20):
    def closure():
        optimizer.zero_grad()
        out = model(inputs)
        loss = criterion(out,outputs)
        loss.backward()
        return loss
    criterion = eval('nn.' + Loss_Function + '()')
    optimizer = optim.LBFGS(model.parameters())
    if Val_input.size() and Val_output.size():
        for epoch in range(num_epochs):
            if use_gpu:
                inputs = Variable(Input).cuda()
                outputs = Variable(Output).cuda()
                val_inputs = Variable(Val_input).cuda()
                val_outputs = Variable(Val_output).cuda()
            else:       
                inputs = Variable(Input)
                outputs = Variable(Output)
                val_inputs = Variable(Val_input)
                val_outputs = Variable(Val_output)
            #forward
            out = model(inputs)
            val_out = model(val_inputs)
            loss = criterion(out, outputs)
            val_loss = criterion(val_out, val_outputs)
            #backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step(closure)
            if (epoch + 1) % exp_step == 0:
                print('Epoch[{}/{}], loss: {:.12f}, val_loss:{:.12f}'
                  .format(epoch + 1, num_epochs, loss.data[0], val_loss.data[0]))
        return model, loss, val_loss
    else:
        for epoch in range(num_epochs):
            if use_gpu:
                inputs = Variable(Input).cuda()
                outputs = Variable(Output).cuda()
            else:       
                inputs = Variable(Input)
                outputs = Variable(Output)
            # Forward
            out = model(inputs)
            loss = criterion(out, outputs)
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step(closure)
            if (epoch + 1) % exp_step == 0:
                print('Epoch[{}/{}], loss: {:.12f}'.format(epoch + 1, num_epochs, loss.data[0]))
        return model, loss

if __name__ == '__main__':
    # Arguements
    path = '/home/pil/Desktop/DC/DC_Reactor.xlsx'
    Hidden_Layer_Structure = [300, 300, 300, 300]
    Activation = 'relu'
    use_gpu = True
    Validation_Ratio = 0.1
    # Reading Datas from DataBase.xlsx
    [Reactor_input, Reactor_output], [Val_Reactor_input, Val_Reactor_output], [Bound_input, Bound_output] = \
    Data_Reading(path, Validation_Ratio = Validation_Ratio, Normalization = True)
    # Define the Loss and transfer the model into gpu
    D_Inputs, D_Outputs = len(Bound_input), len(Bound_output)
    model = Full_Connected(D_Inputs, D_Outputs, Hidden_Layer_Structure, Activation = Activation).double()
    if use_gpu:
        model = model.cuda()
    # Test of the SGD and LBFGS methods
    model, loss = SGD_method(model, Reactor_input, Reactor_output, 
                             Val_Reactor_input, Val_Reactor_output, learning_rate = 1e-4, 
                             num_epochs = 100, use_gpu = use_gpu, Loss_Function = 'MSELoss', exp_step = 20)
    model, loss = SGD_method(model, Reactor_input, Reactor_output, 
                             Val_Reactor_input, Val_Reactor_output, learning_rate = 1e-4, 
                             num_epochs = 100, use_gpu = use_gpu, Loss_Function = 'MSELoss', exp_step = 20)
    model, loss = LBFGS_method(model, Reactor_input, Reactor_output, 
                               Val_Reactor_input, Val_Reactor_output, num_epochs = 100, 
                               use_gpu = use_gpu, Loss_Function = 'MSELoss', exp_step = 20)