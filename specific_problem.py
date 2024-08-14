##################################
import numpy as np
from optimization_algorithm import *
x0 = np.zeros(2)


########## Target function ##########

# In f we let the user to determine the constants in the problem, if not the preset will be as the following
def f(vector, a=1, b=1, c=1):
    if vector[0]<0:
        return 100
    if vector[1]<0:
        return 100
    return -(a * vector[0])** 0.5 -(b * vector[1])**0.5- c * (vector[0] * vector[1])**0.5

########## Constraints ##########
# first constraint -U #

def g1(vector):
    return vector[0]+vector[1]-17

def g2(vector):
    return vector[0]-vector[1]

def g3(vector):
    return (-1) * (vector[0])

def g4(vector):
    return (-1) * (vector[1])

def g(vector):
    return [g1(vector), g2(vector), g3(vector), g4(vector)]

x_solution, f_solution = minimize(f, g, x0)