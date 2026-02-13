import time
import numpy as np
import ferray as tnp

# Test square matrices
print("=== Square Matrices (1024x1024) ===")
a_sq = np.random.rand(1024, 1024).astype(np.float32).tolist()
b_sq = np.random.rand(1024, 1024).astype(np.float32).tolist()

start = time.time()
_ = np.matmul(np.array(a_sq, dtype=np.float32), np.array(b_sq, dtype=np.float32))
np_sq_time = time.time() - start
print(f"NumPy: {np_sq_time:.4f}s")

a_tnp = tnp.NdArray.from_list(a_sq)
b_tnp = tnp.NdArray.from_list(b_sq)
start = time.time()
_ = a_tnp @ b_tnp
tnp_sq_time = time.time() - start
print(f"ferray: {tnp_sq_time:.4f}s")
print(f"Ratio (tnp/np): {tnp_sq_time / np_sq_time:.2f}x slower")

# Test non-square matrices
print("\n=== Non-Square Matrices (512x2048 @ 2048x512) ===")
a_rect = np.random.rand(512, 2048).astype(np.float32).tolist()
b_rect = np.random.rand(2048, 512).astype(np.float32).tolist()

start = time.time()
_ = np.matmul(np.array(a_rect, dtype=np.float32), np.array(b_rect, dtype=np.float32))
np_rect_time = time.time() - start
print(f"NumPy: {np_rect_time:.4f}s")

a_tnp = tnp.NdArray.from_list(a_rect)
b_tnp = tnp.NdArray.from_list(b_rect)
start = time.time()
_ = a_tnp @ b_tnp
tnp_rect_time = time.time() - start
print(f"ferray: {tnp_rect_time:.4f}s")
print(f"Ratio (tnp/np): {tnp_rect_time / np_rect_time:.2f}x slower")
