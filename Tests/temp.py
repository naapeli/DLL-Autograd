import DLL


n = 3
x = DLL.randn((n, n, n))
print(x)
print(x[0])
print(x[1:3])
print(x[0, 1, 1])
print(x[0, 1:3, 0])
