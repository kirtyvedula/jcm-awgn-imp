import torch
import torch.nn as nn
from torch.autograd import Function

# class BinaryLayer(Function):
#     def forward(self, input):
#         return torch.sign(input)
#
#     def backward(self, grad_output):
#         input = self.saved_tensors
#         grad_output[input>1]=0
#         grad_output[input<-1]=0
#         return grad_output
#
# input = torch.randn(4,4)
#
#
# model = BinaryLayer()
# output = model(input)
# loss = output.mean()
#
# loss.backward()
# print(input)
# print(input.grad)

# Erasure channel
import numpy as np
bec_prob = 0.6
noise_shape = [100,7]
x = torch.from_numpy(np.random.choice([0.0, 1.0], noise_shape, p=[bec_prob, 1 - bec_prob])).type(torch.FloatTensor)

import numpy as np
bec_prob = 0.6
noise_shape = [100,7]

def remove_probabilistically(matrix, percent):
    new_array = np.nan([100,6])
    for row in matrix:
        for j in row:
            rand = np.random.randint(
            if rand < percent:
                new_row.append(j)
    new_array =
    return new_array

array1 = np.random.choice([0.0, 1.0], noise_shape, p=[bec_prob, 1 - bec_prob])
array2 = remove_probabilistically(array1, bec_prob)