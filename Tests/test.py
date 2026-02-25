from DLL import Tensor, zeros, ones


a = Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])
b = 1.0 + zeros([2, 3])
c = ones((2, 3))

print("--- Basic Integrity ---")
print(f"A data check: {a.data == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]}")
print(f"B data check: {b.data == c.data}")

print("\n--- Addition ---")
print(f"Tensor + Tensor: {(a + b).data == [2.0, 3.0, 4.0, 5.0, 6.0, 7.0]}")
print(f"Scalar + Tensor: {(1.0 + a).data == [2.0, 3.0, 4.0, 5.0, 6.0, 7.0]}")

print("\n--- Subtraction ---")
print(f"Tensor - Tensor: {(a - b).data == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]}")
print(f"Scalar - Tensor: {(10.0 - a).data == [9.0, 8.0, 7.0, 6.0, 5.0, 4.0]}")

print("\n--- Multiplication ---")
print(f"Tensor * Tensor: {(a * b).data == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]}")
print(f"Tensor * Scalar: {(a * 2.0).data == [2.0, 4.0, 6.0, 8.0, 10.0, 12.0]}")

print("\n--- Division ---")
print(f"Tensor / Scalar: {(a / 2.0).data == [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]}")
# Test 4.0 / [1.0, 2.0, 4.0] = [4.0, 2.0, 1.0]
d = Tensor([1.0, 2.0, 4.0], [3])
print(f"Scalar / Tensor: {(4.0 / d).data == [4.0, 2.0, 1.0]}")

print("\n--- Power ---")
# [1, 2, 3] ** 2.0 = [1, 4, 9]
e = Tensor([1.0, 2.0, 3.0], [3])
print(f"Tensor ** Scalar: {(e ** 2.0).data == [1.0, 4.0, 9.0]}")
# 2.0 ** [1, 2, 3] = [2, 4, 8]
print(f"Scalar ** Tensor: {(2.0 ** e).data == [2.0, 4.0, 8.0]}")
# [2, 2, 2] ** [1, 2, 3] = [2, 4, 8]
f = Tensor([2.0, 2.0, 2.0], [3])
print(f"Tensor ** Tensor: {(f ** e).data == [2.0, 4.0, 8.0]}")
