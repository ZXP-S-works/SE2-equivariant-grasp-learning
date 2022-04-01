
import os
import torch
import torch.nn.functional as F
from e2cnn import gspaces
from e2cnn import nn
import numpy as np


act_d8 = gspaces.Rot2dOnR2(8)
layer1 = nn.R2Conv(nn.FieldType(act_d8, 1 * [act_d8.quotient_repr(2)]),
                   nn.FieldType(act_d8, 1 * [act_d8.quotient_repr(2)]),
                   1)
print("D4/F weights: {}".format(layer1.basisexpansion.dimension()))
layer1.export()

act_c8 = gspaces.Rot2dOnR2(8)
layer2 = nn.R2Conv(nn.FieldType(act_c8, 1 * [act_c8.regular_repr]),
                   nn.FieldType(act_c8, 1 * [act_c8.regular_repr]),
                   1)
print("C4 weights: {}".format(layer2.basisexpansion.dimension()))
