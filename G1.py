import tensorflow as tf
a = tf.constant([[1,2],[3,4]])
b = tf.constant([[5,6],[7,8]])

print("Add:", a+b, "\nMul:", a*b, "\nMatMul:", tf.matmul(a,b))

print("Reshape:", tf.reshape(a,[4]), " Slice:", b[:,0])

@tf.function
def hyp(x,y):
    return tf.sqrt(x**2 + y**2)

print("Graph Result:", hyp(3.,4.))

for i in range(1,4): 
    print("Square:", tf.square(i).numpy())

print("Broadcast:", tf.constant([1,2,3])+tf.constant([[1],[2],[3]]))


#---------------------------------------------

import tensorflow as tf

print("TF Version:", tf.__version__)
print("Eager Execution:", tf.executing_eagerly())

# Tensor creation
a = tf.constant([[1, 2], [3, 4]])
b = tf.constant([[5, 6], [7, 8]])

# Basic operations
print("Add:\n", a + b)
print("Multiply:\n", a * b)
print("MatMul:\n", tf.matmul(a, b))

# Reshape and slice
print("Reshaped A:", tf.reshape(a, [4]))
print("Slice B first col:", b[:, 0])

# Computation graph
@tf.function
def hypotenuse(x, y):
    return tf.sqrt(x**2 + y**2)

print("Hypotenuse (graph):", hypotenuse(3.0, 4.0))

# Eager execution example
for i in range(1, 4):
    print(f"Square of {i}:", tf.square(tf.constant(i)).numpy())

# Broadcasting
v = tf.constant([1, 2, 3])
m = tf.constant([[1], [2], [3]])
print("Broadcasting result:\n", v + m)
