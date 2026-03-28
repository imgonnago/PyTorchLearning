import torch
import numpy as np
from sympy import print_tree

#Directly from data
data = [[1,2],[3,4]]
x_data = torch.tensor(data)

#From a Numpy array
np_array = np.array(x_data)
x_np = torch.from_numpy(np_array)

#From another tensro
x_ones = torch.ones_like(x_data) # retains the properties of x_data
print(f'Ones Tensor: \n {x_ones} \n')

x_rand = torch.rand_like(x_data, dtype=torch.float) # overrides the datatype of x_data
print(f'Random Tnesor: \n {x_rand} \n')

#With random or constant values:
shape = (2,3,)
rand_tensor = torch.rand(shape)
ones_tensror = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f'Random tensor: \n {rand_tensor} \n')
print(f'Ones tensro: \n {ones_tensror} \n')
print(f'Zeros tensor: \n {zeros_tensor} \n')

#Attribute of a Tensor
tensor = torch.rand(3,2)

print(f'tensor shape: {tensor.shape}')
print(f'tensor dtype: {tensor.dtype}')
print(f'tensor device: {tensor.device}')

#Operation on tensors
"""By default, tensors are created on the CPU. 
We need to explicitly move tensors to the accelerator using .to method (after checking for accelerator availability). 
Keep in mind that copying large tensors across devices can be expensive in terms of time and memory!"""

#We move our tensor to the current accelerator if available
if torch.accelerator.is_available():
    tensor = tensor.to(torch.accelerator.current_accelerator())

#Standard numpy-like indexing and slicing:

tensor = torch.rand(4,4)
print(f'Fist row: {tensor[0]}')
print(f'Fist column: {tensor[:,0]}')
print(f'Last column: {tensor[:,-1]}')
tensor[:,1] = 0
print(tensor)

#Joining tensors. tou can use torch.stack also. but stack makes another dimension
t1 = torch.cat([tensor,tensor,tensor], dim = 1)
print(t1)

#Arithmetic operations
#This computes the matrix multiplication between two tensors. y1, y2, y3 will have the same value
#''tensor.T'' returns the transpose of a tensor
y1  = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)

y3 = torch.rand_like(y1)
torch.matmul(tensor, tensor.T, out=y3)

#This computes the element-wise product. z1, z2, z3 will have the same value
z1 = tensor * tensor
z2 = tensor.mul(tensor)

z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)

#Single-element tensors
agg = tensor.sum()
agg_item = agg.item()
print(agg_item, type(agg_item))

#In-place operation
"""
In-place operations save some memory, 
but can be problematic when computing derivatives because of an immediate loss of history. 
Hence, their use is discouraged.
"""
print(f'{tensor} \n')
tensor.add_(5)
print(f'{tensor}')

#Tensor to Numpy array

t = torch.ones(5)
print(f't: {t}')
n = t.numpy()
print(f'n: {n}')
#A Changes in the tensor reflects in the Numpy array.
t.add_(5)
print(f't: {t}')
print(f'n: {n}')

n = np.ones(5)
t = torch.from_numpy(n)
print(f'n: {n}')
print(f't: {t}')

np.add(n,1, out=n)
print(f'n: {n}')
print(f't: {t}')