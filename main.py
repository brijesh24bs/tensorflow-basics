import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

# Initialization of Tensors
x = tf.constant(4, shape=(1,1), dtype=tf.float32)
x = tf.constant([[1,2,3],
                 [4,5,6]])

x = tf.ones((2,3), dtype=tf.int32)
x = tf.zeros(shape=(3,3), dtype=tf.float32)
x = tf.eye(3) # I for the Identity matrix

x = tf.random.normal((3,3), mean=0, stddev=1)
x = tf.random.uniform((1,3) , minval=0, maxval=1)
x = tf.range(9)
x = tf.range(start=1, limit=10, delta=2)
x = tf.cast(x, dtype=tf.float64)



# Mathematical Operations
x = tf.constant([1,2,3])
y = tf.constant([9,8,7])
z = tf.add(x, y)
z = x + y

z = tf.subtract(x, y)
z = x - y

z = tf.divide(x, y)
z = x / y

z = tf.multiply(x, y)
z = x * y

z = tf.tensordot(x, y, axes=1)
z = tf.reduce_sum(x*y, axis=0)
# print(x*y)
# print(z)

z = x**5

x = tf.random.normal((2,3))
y = tf.random.normal((3,4))
z = tf.matmul(x,y) # Same as z = x @ y
# print(z)

# Indexing of Tensor
x = tf.constant([0, 1, 2, 8, 1, 2, 3, 8, 9, 100])
# print(x[:])
# print(x[1:])
# print(x[1:3])
# print(x[1::3])

# print(x[::-1])
indices = tf.constant([0, 3])
print(tf.gather(x, indices))

x = tf.constant([[1, 2],
                 [3, 4],
                 [5, 6]])
# print(x[1:, 1:])


# Reshaping
x = tf.range(9)
print(x)
x = tf.reshape(x, (3,3))
print(x)

x = tf.transpose(x, perm=[1,0])
