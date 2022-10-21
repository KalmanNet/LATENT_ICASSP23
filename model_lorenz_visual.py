## model lorenz visual ##
import math
import torch
from numpy.linalg import matrix_power
torch.pi = torch.acos(torch.zeros(1)).item() * 2  # which is 3.1415927410125732
from torch import autograd
#from filing_paths import path_model
import sys
import numpy as np
import torch.nn as nn
import warnings
#suppress warnings
warnings.filterwarnings('ignore')

if torch.cuda.is_available():
    dev = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    dev = torch.device("cpu")
    #print("Running on the CPU")
################for encoded transition ###################
y_size = 28
encoded_dimention = 3
class h_latent_space(nn.Module):
    def __init__(self, weights, bias, d,m):
        super(h_latent_space, self).__init__()
        self.fc = nn.Linear(m, d)
        with torch.no_grad():
            self.fc.weight.copy_(weights)
            self.fc.bias.copy_(bias.reshape(bias.shape[0]))
        # self.fc2 = nn.Linear(10, 5)

    def forward(self, x):
        x1 = self.fc(x.double())
        # x2 = self.fc2(x1)
        return x1
############# for EKF #####################
#### dinamic process F matrix
eps = 1e-8
m =3
#50,
m1x_0 = torch.ones(m, 1)
m2x_0 = 0 * 0 * torch.eye(m)
#### obzevation process H matrix
H = torch.eye(3)
b = torch.tensor([[0.0],
                  [0.0],
                 [0.0]])
n = y_size*y_size
###### for sys model #####

# def f(x):
#     delta=0.02
#     x1, x2, x3 = x  # Unpack the state vector
#     A = torch.tensor([[-10.0, 10.0, 0.0],
#                       [28.0, -1.0, -x1],
#                       [0.0, x1, -8/3]])
#     A_matrixes_list = [matrix_power(delta*A, j)/math.factorial(j) for j in range(5)]
#     F= torch.eye(3)+np.sum(A_matrixes_list)
#     return torch.matmul(F, x)

eps = 1e-8
delta_t = 0.02
J = 5
B = torch.tensor([[[0,  0, 0],[0, 0, -1],[0,  1, 0]], torch.zeros(m,m), torch.zeros(m,m)]).float()
C = torch.tensor([[-10, 10,    0],
                  [ 28, -1,    0],
                  [  0,  0, -8/3]]).float()
#,N_samples
#x_all
def F(x):
    # A = torch.add(torch.einsum('nhw,wa->nh', B, x).T,C)
    #F_all=torch.empty(N_samples, m, m)
    #for i in range(N_samples):
        #x=x_all[i]
    A = (torch.add(torch.reshape(torch.matmul(B, x.float()), (m, m)).T, C)).to(dev)
    # Taylor Expansion for F
    F = torch.eye(m)
    for j in range(1, J + 1):
        F_add = (torch.matrix_power(A * delta_t, j) / math.factorial(j)).to(dev)
        F = torch.add(F, F_add).to(dev)
        #F_all[i,:,:]=F
    return F.double()

def h(x):
    x1 = np.round(x[0].item(),3)
    x2 = np.round(x[1].item(),3)
    x3 = np.round(x[2].item(),3)
    sigma=0.2
    y_sample = torch.zeros(y_size, y_size).double()
    for i in range(y_size):
        for j in range(y_size):
            mone = np.round(np.math.pow((i-x1),2)+math.pow((j-x2),2),3)
            mechane = 2*x3
            y_sample[i, j] =np.round(10*np.exp(-mone/mechane),3)
    return y_sample

def get_h_derivative(x):
    x1 = np.round(x[0].item(),3)
    x2 = np.round(x[1].item(),3)
    x3 = np.round(x[2].item(),3)
    #sigma=0.2
    H_drivative = torch.zeros(y_size, y_size, m).double()
    for i in range(y_size):
        for j in range(y_size):
            mone = np.round(math.pow((i-x1),2)+math.pow((j-x2),2),3)
            mechane = 2*x3
            exp = np.round(10*np.exp(-mone/mechane),3)
            H_drivative[i, j, 0]= np.round(exp*(i-x1)/x3,3)
            H_drivative[i, j, 1] = np.round(exp*(j-x2)/x3,3)
            H_drivative[i, j, 2] = np.round(exp*mone/(2*math.pow(x3,2)),3)
    return H_drivative
##########################
#
# def f_test(x):
#     # A = torch.add(torch.einsum('nhw,wa->nh', B, x).T,C)
#     A = torch.add(torch.reshape(torch.matmul(B, x), (m, m)).T, C)
#
#     # Taylor Expansion for F
#     F = torch.eye(m)
#     for j in range(1, J + 1):
#         F_add = (torch.matrix_power(A * delta_t_test, j) / math.factorial(j)).to(dev)
#         F = torch.add(F, F_add).to(dev)
#
#     return torch.matmul(F, x)
#
#
# def f_gen(x):
#     # A = torch.add(torch.einsum('nhw,wa->nh', B, x).T,C)
#     A = torch.add(torch.reshape(torch.matmul(B, x), (m, m)).T, C)
#
#     # Taylor Expansion for F
#     F = torch.eye(m)
#     for j in range(1, J + 1):
#         F_add = (torch.matrix_power(A * delta_t_gen, j) / math.factorial(j)).to(dev)
#         F = torch.add(F, F_add).to(dev)
#
#     return torch.matmul(F, x)
#
# def fInacc(x):
#     # A = torch.add(torch.einsum('nhw,wa->nh', B, x).T,C)
#     A = torch.add(torch.reshape(torch.matmul(B_mod, x), (m, m)).T, C_mod)
#
#     # Taylor Expansion for F
#     F = torch.eye(m)
#     for j in range(1, J_mod + 1):
#         F_add = (torch.matrix_power(A * delta_t_mod, j) / math.factorial(j)).to(dev)
#         F = torch.add(F, F_add).to(dev)
#
#     return torch.matmul(F, x)
#
#
# def fRotate(x):
#     A = (torch.add(torch.reshape(torch.matmul(B, x), (m, m)).T, C)).to(dev)
#     A_rot = torch.mm(RotMatrix, A)
#     # Taylor Expansion for F
#     F = torch.eye(m)
#     for j in range(1, J + 1):
#         F_add = (torch.matrix_power(A_rot * delta_t, j) / math.factorial(j)).to(dev)
#         F = torch.add(F, F_add).to(dev)
#
#     return torch.matmul(F, x)

#
# def hInacc(x):
#     return torch.matmul(H_mod, x)
#     # return toSpherical(x)
#
#
# def h_nonlinear(x):
#     return toSpherical(x)
#
#
# def getJacobian(x, a):
#     # if(x.size()[1] == 1):
#     #     y = torch.reshape((x.T),[x.size()[0]])
#     try:
#         if (x.size()[1] == 1):
#             y = torch.reshape((x.T), [x.size()[0]])
#     except:
#         y = torch.reshape((x.T), [x.size()[0]])
#
#     if (a == 'ObsAcc'):
#         g = h
#         print("h function")
#     elif (a == 'ModAcc'):
#         g = f
#         print("f function")
#     elif (a == 'ObsInacc'):
#         g = hInacc
#     elif (a == 'ModInacc'):
#         g = fInacc
#
#     Jac = autograd.functional.jacobian(g, y)
#     Jac = Jac.view(-1, m)
#     return Jac
#
#
# def toSpherical(cart):
#     rho = torch.norm(cart, p=2).view(1, 1)
#     phi = torch.atan2(cart[1, ...], cart[0, ...]).view(1, 1)
#     phi = phi + (phi < 0).type_as(phi) * (2 * torch.pi)
#
#     theta = torch.acos(cart[2, ...] / rho).view(1, 1)
#
#     spher = torch.cat([rho, theta, phi], dim=0)
#
#     return spher
#
#
# def toCartesian(sphe):
#     rho = sphe[0, :]
#     theta = sphe[1, :]
#     phi = sphe[2, :]
#
#     x = (rho * torch.sin(theta) * torch.cos(phi)).view(1, -1)
#     y = (rho * torch.sin(theta) * torch.sin(phi)).view(1, -1)
#     z = (rho * torch.cos(theta)).view(1, -1)
#
#     cart = torch.cat([x, y, z], dim=0)
#
#     return cart
#
#
# def hInv(y):
#     return torch.matmul(H_design_inv, y)
#     # return toCartesian(y)
#
#
# def hInaccInv(y):
#     return torch.matmul(H_mod_inv, y)
#     # return toCartesian(y)

# x = torch.tensor([[1],[1],[1]]).float()
# H = getJacobian(x, 'ObsAcc')
# print(H)
# print(h(x))
# F = getJacobian(x, 'ModAcc')
# print(F)
# print(f(x))