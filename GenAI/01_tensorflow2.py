import tensorflow as tf

# Check if eager execution is enabled
print("Eager Execution Enabled:", tf.executing_eagerly())

print("\n=== 1. Tensor Creation ===")

# Creating different types of tensors
scalar = tf.constant(10)
vector = tf.constant([1, 2, 3, 4])
matrix = tf.constant([[1, 2], [3, 4]])
tensor3d = tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

print("Scalar:\n", scalar)
print("Vector:\n", vector)
print("Matrix:\n", matrix)
print("3D Tensor:\n", tensor3d)

# Creating tensors with specific values
zeros_tensor = tf.zeros([2, 3])
ones_tensor = tf.ones([2, 3])
random_tensor = tf.random.uniform([2, 3], minval=0, maxval=10, dtype=tf.int32)

print("\nTensor of Zeros:\n", zeros_tensor)
print("Tensor of Ones:\n", ones_tensor)
print("Random Tensor (0–9):\n", random_tensor)

print("\n=== 2. Tensor Manipulation ===")

# Reshaping tensors
reshaped_tensor = tf.reshape(random_tensor, [3, 2])
print("Original Shape:", random_tensor.shape)
print("Reshaped Tensor:\n", reshaped_tensor)

# Slicing and Indexing
print("\nFirst row of random_tensor:\n", random_tensor[0])
print("Element at [0,1]:", random_tensor[0, 1].numpy())

# Broadcasting Example
a = tf.constant([[1], [2], [3]])
b = tf.constant([4, 5, 6])
broadcast_sum = a + b
print("\nBroadcasting Example (a + b):\n", broadcast_sum)

# Basic Arithmetic Operations
x = tf.constant([[5, 10], [15, 20]])
y = tf.constant([[1, 2], [3, 4]])
add_result = tf.add(x, y)
mul_result = tf.multiply(x, y)
matmul_result = tf.matmul(x, y)

print("\nAddition Result:\n", add_result)
print("Multiplication Result (element-wise):\n", mul_result)
print("Matrix Multiplication Result:\n", matmul_result)

print("\n=== 3. Eager Execution Demonstration ===")

# Demonstrating dynamic computation with Eager Execution
for i in range(1, 4):
    val = tf.constant(i * 5)
    result = tf.square(val)
    print(f"Iteration {i}: Value = {val.numpy()}, Square = {result.numpy()}")

# Using tf.Variable for dynamic computation
var = tf.Variable(5.0)
print("\nInitial Variable:", var.numpy())

for step in range(3):
    var.assign_add(2.0)
    print(f"After Step {step+1}, Variable Value = {var.numpy()}")

print("\n All tensor operations executed dynamically using Eager Execution.")
