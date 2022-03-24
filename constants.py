import os
import torch
import numpy as np

#============================== 
# General
#============================== 
# model \in {'dielectric', 'neohookean', 'gent', 'ArrudaBoyce', 'Ogden'}
model = 'dielectric'    # \in {'dielectric', 'neohookean', 'gent', 'ArrudaBoyce', 'Ogden'}
material = 'VHB-1'
#material = 'VHB-2'

#============================== 
# Actuator
#============================== 
# Mechanical Properties;
if material=='VHB-1':
    eps = 4.758e-11         # permittivity (As/Vm); Mat.1
elif material=='VHB-2':
    eps = 4.1595-11         # permittivity (As/Vm); Mat.2
Y = 1.84e7        # Young's Modulus, 400% biaxially prestrained (Pa)

# Structure
a_0 = 0.004     # Radius of the electrode (m)
#pre = 4.0         # pre-stretch
pre = 1/16         # pre-stretch
#z_0 = 6.25e-5     # Original thickness, 0.001/16 (m)
z_0 = 0.001     # Original thickness, 0.001 (m)
#rho = 960         # Density (kg/m^3)
rho = 1200         # Density (kg/m^3)
#anal_sr = 384000
anal_sr = 48000
save_sr = 48000
t_sup = 0.1    # TODO: numerator underflow?
f_sup = 20000
f_delta = 2000
do_filter = True

#============================== 
# Models
#============================== 
# Input Parameters
Vdc = 1.000e3     # DC Bias (V)
Vpp = 0.500e3     # AC pk-pk (V)

# Hyper-elastic model
#mu = 6e5    # shear modulus (Pa)
if material=='VHB-1':
    mu = 16966      # shear modulus (Pa); Mat.1
    beta = 2.333    # Mat.1
    tau = 72.377    # (sec); Mat.1
elif material=='VHB-2':
    mu = 22800    # shear modulus (Pa); Mat.2
    beta = 2.85   # Mat.2
    tau = 200     # (sec); Mat.2

def smoothe(t, T, func='sinsin'):
    if t < T:
        if func=='ramp':
            return t/T
        elif func=='sinlin':
            return np.sin((np.pi/2) * (t/T))**2
        elif func=='sinsqrt':
            return np.sin((np.pi/2) * ((t**.5)/T))**2
        elif func=='sinsin':
            return np.sin((np.pi/2) * np.sin((np.pi/2) * (t/T)))**2
        else:    # id
            return 1
    else:
        return 1

