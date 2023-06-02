import PyBuddy

sizes_l = [3]
t1_memref = PyBuddy.MemRef1d([1.0, 2.0, 3.0], sizes_l, 0)
t2_memref = PyBuddy.MemRef1d([11.0, 5.0, 0.0], sizes_l, 0)
result_memref = PyBuddy.MemRef1d([0.0, 0.0, 0.0], sizes_l, 0)
PyBuddy.arith_add(result_memref, t1_memref, t2_memref)
print(result_memref.getData())