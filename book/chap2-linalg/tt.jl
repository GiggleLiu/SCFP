Slide Type
Slide
A simple implementation of the MPS Born Machine
Pan Zhang
Institute of Theoretical Physics, Chinese Academy of Sciences
Reference: Z. Han, J. Wang, H. Fan, L. Wang and P. Zhang, Phys. Rev. X 8, 031012 (2018)

import sys
import numpy as np
import torch 
import math,time
%matplotlib inline
import matplotlib.pyplot as plt

torch.manual_seed(1) # Fix seed of the random number generators
np.random.seed(1)

Data loading
1000
  MNIST images have been stored as "mnist784_bin_1000.npy".
Each image contains  𝑛=28×28=784
  pixels, each of which takes value  0
  or  1
 .
Each image is viewed as a product state in the Hilbert space of dimension  2𝑛.

n=784 # number of qubits
#m=1000 # m images
m=20
data=np.load("mnist784_bin_1000.npy").astype(np.int32)
data=data[:m,:]
data=torch.LongTensor(data)