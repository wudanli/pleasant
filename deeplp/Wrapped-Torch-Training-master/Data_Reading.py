import pandas as pd
import torch
import copy
import numpy as np
import math

# Data_Xlsx To Tensor
def Data_Reading(record_path, Validation_Ratio = 0.15, Normalization = True):
    # Read the xlsx
    Input = pd.read_excel(record_path, sheetname = 'Input')
    Output = pd.read_excel(record_path, sheetname = 'Output')
    # Store the bounds
    Input_bounds = [[Input[col].max(), Input[col].min()] for col in Input]
    Output_bounds = [[Output[col].max(), Output[col].min()] for col in Output]
    # Normalization
    Input_Normed = copy.deepcopy(Input).apply(lambda x : (x - np.min(x)) / (np.max(x) - np.min(x)))
    Output_Normed = copy.deepcopy(Output).apply(lambda x : (x - np.min(x)) / (np.max(x) - np.min(x)))
    if Normalization:
        Input = torch.from_numpy(Input_Normed.values)
        Output = torch.from_numpy(Output_Normed.values)
    else:
        Input = torch.from_numpy(Input.values)
        Output = torch.from_numpy(Output.values)
    if Validation_Ratio:
        return [Input [0 : Input.size()[0] - math.ceil(Input.size()[0] * Validation_Ratio), :], 
                Output[0 : Input.size()[0] - math.ceil(Input.size()[0] * Validation_Ratio), :]],\
               [Input [Input.size()[0] - math.ceil(Input.size()[0] * Validation_Ratio) : , :],
                Output[Input.size()[0] - math.ceil(Input.size()[0] * Validation_Ratio) : , :]],\
               [Input_bounds, Output_bounds]
    else:
        return [Input, Output], [torch.Tensor(0,0), torch.Tensor(0,0)], [Input_bounds, Output_bounds]

if __name__ == '__main__':
    # Test of Data_Reading
    reactor_path = '/home/pil/Desktop/DC/DC_Reactor.xlsx'
    seperator_path = '/home/pil/Desktop/DC/DC_Seperator.xlsx'
    [Reactor_input, Reactor_output], [Val_Reactor_input, Val_Reactor_output],\
    [Bound_input_1, Bound_output_1] = Data_Reading(reactor_path, Validation_Ratio = 0, Normalization = False)
    [Seperator_input, Seperator_output], [Val_Seperator_input, Val_Seperator_output],\
    [Bound_input_2, Bound_output_2] = Data_Reading(seperator_path, Normalization = True)