# Set up
import numpy as np
# Scalar
x = np.array(5)
# Vector
x = np.array([4 , 2 , 1])
# Matrix
x = np.array([[1,2], [3,4]])
# 3-D Tensor
x = np.array([[[1,2],[3,4]],[[5,6],[7,8]]])
# Functions
print ("np.zeros((3,3)):\n", np.zeros((3,3)))
print ("np.ones((2,2)):\n", np.ones((2,2)))
print ("np.eye((3)):\n", np.eye((3))) # identity matrix
#Indexing
x = np.array([1, 2, 3])
print ("x: ", x)
print ("x[0]: ", x[0])
# Slicing
x = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
print (x)
print ("x column 1: ", x[:, 1])
# Arithmetic
x = np.array([[1,2], [3,4]], dtype=np.float64)
y = np.array([[1,2], [3,4]], dtype=np.float64)
print ("x + y:\n", np.add(x, y)) 
print ("x - y:\n", np.subtract(x, y)) 
print ("x * y:\n", np.multiply(x, y)) 
# Dot product
a = np.array([[1,2,3], [4,5,6]], dtype=np.float64) 
b = np.array([[7,8], [9,10], [11, 12]], dtype=np.float64)
c = a.dot(b)
# Axis operations
# Sum across a dimension
x = np.array([[1,2],[3,4]])
print (x)
print ("sum all: ", np.sum(x)) # adds all elements
print ("sum axis=0: ", np.sum(x, axis=0)) # sum across rows
print ("sum axis=1: ", np.sum(x, axis=1)) # sum across columns
# Reshape
x = np.array([[1,2,3,4,5,6]])
print (x)
print ("x.shape: ", x.shape)
y = np.reshape(x, (2, 3))
print ("y: \n", y)
print ("y.shape: ", y.shape)
