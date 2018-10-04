from Data_Reading import Data_Reading
from Networks import Full_Connected
from Methods import LBFGS_method
from Evaluate import Evaluate
from tools import Recover

# Arguements
path = '/home/pil/Desktop/DC/DC_Reactor.xlsx'
Hidden_Layer_Structure = [100, 100, 100]
Activation = 'relu'
use_gpu = True
Validation_Ratio = 0.2
# Reading Datas from DataBase.xlsx
[Reactor_input, Reactor_output], [Val_Reactor_input, Val_Reactor_output], [Bound_input, Bound_output] = \
Data_Reading(path, Validation_Ratio = Validation_Ratio, Normalization = True)
# Define the Loss and transfer the model into gpu
D_Inputs, D_Outputs = len(Bound_input), len(Bound_output)
model = Full_Connected(D_Inputs, D_Outputs, Hidden_Layer_Structure, Activation = Activation).double()
if use_gpu:
    model = model.cuda()
# Train
model, loss, val_loss = LBFGS_method(model, Reactor_input, Reactor_output, 
                           Val_Reactor_input, Val_Reactor_output, num_epochs = 1000, 
                           use_gpu = use_gpu, Loss_Function = 'MSELoss', exp_step = 1)
# Evaluate the model
predict, final_loss = Evaluate(model, Reactor_input, Reactor_output,
                               Val_Reactor_input, Val_Reactor_output, 
                               use_gpu = use_gpu)
# Recover from Normalization
Predicted_output = Recover(predict, Bound_output)