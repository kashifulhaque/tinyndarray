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
