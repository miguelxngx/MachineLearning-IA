from keras.datasets import mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

print("Train images number of dimensions: ", train_images.ndim)
print("Train images shape: ", train_images.shape)
print("Train images dtype: ", train_images.dtype)

#2.6 Displaying the four digit
#This display the fourth digit  in this 3D tensor, using the library Matplotlib

digit = train_images[4]

import matplotlib.pyplot as plt
plt.imshow(digit, cmap=plt.cm.binary)
plt.show()

#Manipulating tensors in Numpy
#The folloeing example selects digits #10 to #100 (#100 isn't included)
#and puts them in an array of shape (90, 28, 28):
my_slice = train_images[10:100, :, :]
print(my_slice.shape)

#In order to select 14x14 pixels in the bottim-right corner of all images,
#you do this:
my_slice = train_images[:, 14:, 14:]

#In order to crop the images to patches of 14x14 pixels centered in the
# middle, you do this:
my_slice = train_images[:, 7:-7, 7:-7]

#The first axis in all data tensors you'll come accross in deep learning
#will be the samples axis.
#Deep-learning models don't proccess an entire dataset at once; rather,
#they break the data into small batches. Concretely, here's one batch
#of our MNIS digits, with batch size of 128:
batch = train_images[:128]
#and here's the next batch:
batch = train_images[128:256]
#and the nth batch:
#batch = train_images[128 * n:128* (n+1)]

import keras
#A keras instance looks like this:
keras.layers.Dense(512, activation='relu')
#This layer can be interpreted as a function, wich takes as input a 2D
#tensor and returns another 2D tensor-a new representation for the
#input tensor. Specially the function is as follows(where W is a 2D
# tensor and b is a vector, both attributs of the layer):
#output = relu(dot(W, input) + b)

#If you want to write a naive Python implementation of an element-wise 
# operation, you use a for loop, as in this naive implementation of an
# element_wise relu operation:
def naive_relu(x):
    assert len(x.shape) == 2 # x is a 2D Numpy tensor

    x = x.copy() # Avoid overwritting the input tensor
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i, j] = max(x[i, j], 0)
    return x

#You do the same for addition:
def naive_add(x, y):
    assert len(x.shape) == 2 # x and y are 2D Numpy tensors
    assert x.shape == y.shape

    x = x.copy()
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i, j] += y[i, j]
    return x

# On the same principle, you can do element-wise multiplication,
# substracion and so on.

#In practice, when dealing with Numpy arrays, these operations are
# available as welloptimized built-in Numpy functions, which themselves
# delegate the heavy lifting to a Basic Linear Algebra Subprograms (BLAS)
# implementation if you have one installed(which you should). BLAS are
# low-level, highly parallel, efficient tensor-manipulation 
# routines that are typically implemented in Fortran or C.
# So, in Numpy, you can do the following element-wise operation, and it
# will be blazing fast:

#import numpy as np
#z = x + y Element-wise addition
#z = np.maximum(z, 0.) Element-wise relu

#When the shapes of tensors differ, the smaller tensor will be broadcasted
# to match the shape of the larger tensor. Broadcasting consist in 2 steps:
#   1. Axes(called broadcast axes) are added to the smaller tensor to match
#      the larger tensor.
#   2. Te smaller tensor is replaced alongside these new axes to match the
#      full shape of a larger tensor.
# Here's what a naive implementation would look like:

def naive_add_matrix_and_vector(x, y):
    assert len(x.shape) == 2
    assert len(y.shape) == 1
    assert x.shape[1] == y.shape[0]

    x = x.copy()
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i, j] += y[j]
    return x

import numpy as np

x = np.random.random((64, 3, 32, 10)) #x is a random tensor with shape
                                      #(64, 3, 32, 10).
y = np.random.random((32, 10)) #y is a random tensor with shape (32, 10)

z = np.maximum(x, y) # the output z has shape (64, 3, 32, 10) like x.

#The dot operation, also called a tensor product, is the most commin, most
# useful tensor operation, Contrary to element-wise operations, it combines
# entries in the input tensors.
#z = np.dot(x, y)
# In mathematical notation, you'd note the operation with a dot(.):
# z = x . y

def naive_vector_dot(x, y):
    assert len(x.shape) == 1
    assert len(y.shape) == 1
    assert x.shape[0] == y.shape [0]

    z = 0.
    for i in range(x.shape[0]):
        z += x[i] * y[i]
    return z
# The dot product betwween two vectors is a scalar and that only vectors
# with the same number of elements are compatible for a dot product.

# You can also take the dot procut between a matrix x and a vector y, wich
# returns a vector where the coefficients are the dot products between y
# and the rows of x.

def naive_matrix_vector_dot(x,y):
    assert len(x.shape) == 2
    assert len(y.shape) == 1
    assert x.shape[1] == y.shape[0]

    z = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        z[i] = naive_vector_dot(x[i,:], y)
    return z

def naive_matrix_dot(x, y):
    assert len(x.shape) == 2
    assert len(y.shape) == 2
    assert x.shape[1] == y.shape[0]

    z = np.zeros((x.shape[0], y.shape[1]))
    for i in range(x.shape[0]):
        for j in  range(y.shape[1]):
            row_x = x[i, :]
            column_y = y[:, j]
            z[i, j] = naive_vector_dot(row_x, column_y)
    return x

#Tensor reshaping
x = np.array([[0., 1.],
              [2., 3,],
              [4., 5.]])
print(x.shape)

x = x.reshape((6, 1))
print(x)

x = x.reshape((2,3))
print(x)

#A special case of reshapping that's commonly encountered is transposition.
# Transposing a matrix means exchanging its rows and its columns, so that
# x[i, :] becomes x[:, i]:
x = np.zeros((300, 20))
x = np.transpose(x)
print(x.shape)