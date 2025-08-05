import time
import numpy as np
import random
import tinyndarray as tnp

def random_matrix(rows, cols):
  return [[random.random() for _ in range(cols)] for _ in range(rows)]

def time_numpy(a, b):
  start = time.time()
  _np_result = np.matmul(a, b)
  return time.time() - start

def time_tinyndarray(a_list, b_list):
  a = tnp.NdArray.from_list(a_list)
  b = tnp.NdArray.from_list(b_list)
  start = time.time()
  _c = a @ b
  return time.time() - start

def check_correctness(a_list, b_list):
  a_np = np.array(a_list)
  b_np = np.array(b_list)
  c_np = np.matmul(a_np, b_np).tolist()

  a = tnp.NdArray.from_list(a_list)
  b = tnp.NdArray.from_list(b_list)
  c = a @ b
  c_list = c.to_list()

  rel_tol = 1e-5
  abs_tol = 1e-6

  # Use relative tolerance to allow float noise
  for i, (row1, row2) in enumerate(zip(c_np, c_list)):
    for j, (x, y) in enumerate(zip(row1, row2)):
      if abs(x - y) > max(rel_tol * max(abs(x), abs(y)), abs_tol):
        print(f"Mismatch at ({i},{j}): numpy={x:.5f}, rust={y:.5f}")
        return False
  return True

if __name__ == "__main__":
  shape = (200, 300, 400)  # A: 200x300, B: 300x400
  print(f"Running benchmark for shape: {shape[0]}x{shape[1]} @ {shape[1]}x{shape[2]}")

  a_list = random_matrix(shape[0], shape[1])
  b_list = random_matrix(shape[1], shape[2])

  # Correctness check
  print("Checking correctness...", end="")
  if check_correctness(a_list, b_list):
    print("✅ Passed")
  else:
    print("❌ Failed")
    exit(1)

  # Time NumPy
  print("Benchmarking NumPy...")
  np_time = time_numpy(np.array(a_list), np.array(b_list))
  print(f"NumPy time: {np_time:.4f}s")

  # Time tinyndarray
  print("Benchmarking tinyndarray (Rust)...")
  tnp_time = time_tinyndarray(a_list, b_list)
  print(f"tinyndarray time: {tnp_time:.4f}s")

  speedup = np_time / tnp_time
  print(f"Speedup (tinyndarray vs NumPy): {speedup:.2f}x")
