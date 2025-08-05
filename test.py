import numpy as np
import tinyndarray as tnp

a = tnp.NdArray([2, 3])
print("ndim:", a.ndim())
print("shape:", a.shape())
print("before", a.get([0, 1]))

a.set(
  [0, 1],
  42.0
)

print("after:", a.get([0, 1]))
print(a)

print("----------------")
a = tnp.NdArray.zeros([3, 4])
print(a)
print("----------------")

print("a[0, 2] =", a[0, 2])  # ➜ 0.0
a[0, 2] = 99
print("a[0, 2] =", a[0, 2])  # ➜ 99.0
print(a)

print("----- ----- ----- -----")
a = tnp.NdArray.ones([2, 3])
print(a)
print("----- ----- ----- -----")

print("----- ----- ----- -----")
a = tnp.NdArray.ones([2, 3])
print(a.shape())  # ➜ [2, 3]
a.reshape([3, 2])
print(a.shape())  # ➜ [3, 2]
print("----- ----- ----- -----")

print("----- ----- ----- -----")
a = tnp.NdArray.ones([2, 3])
b = tnp.NdArray.ones([1, 3])
c = a.add(b)

print("a:", a)
print("b:", b)
print("c (a + b):", c)
print("----- ----- ----- -----")

print("----- ----- ----- -----")
a = tnp.NdArray.ones([2, 3])
b = tnp.NdArray.ones([1, 3])
c = a + b  # ➜ works via __add__
print(c)
print("----- ----- ----- -----")

print("----- ----- ----- -----")
a = tnp.NdArray.ones([2, 3])
b = tnp.NdArray.ones([1, 3]) * 2.0

print("a + b =", (a + b))
print("a - b =", (a - b))
print("a * b =", (a * b))
print("a / b =", (a / b))
print("----- ----- ----- -----")

print("----- ----- ----- -----")
a = tnp.NdArray.ones([2, 3])
print(a + 2.0)
print(a * 5.0)
print(a / 2.0)
print(a - 1.0)
print("----- ----- ----- -----")

print("----- ----- ----- -----")
a = tnp.NdArray.from_list(
  [
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0]
  ]
)

print("Original:")
print(a)

print("Transposed:")
print(a.transpose())
print("----- ----- ----- -----")

print("----- ----- ----- -----")
a = tnp.NdArray.from_list([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
print("Python list:", a.to_list())

t = a.transpose()
print("Transposed list:", t.to_list())
print("----- ----- ----- -----")

print("----- ----- ----- -----")
print("From Numpy")
a_np = np.ones((2, 3), dtype=np.float32)
a_rust = tnp.NdArray.from_numpy(a_np)

print("To Numpy")
a_back = a_rust.to_numpy()
print(a_back)

a = np.array([[1, 2], [3, 4]], dtype=np.float32)
b = tnp.NdArray.from_numpy(a)
c = b.to_numpy()

print("Original NumPy:\n", a)
print("Back from Rust:\n", c)

print("Equal?", np.allclose(a, c))

print("----- ----- ----- -----")
