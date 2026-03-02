import DLL


DLL.Random.seed(0)

n = 3
x = DLL.randn((n, n, n))
print(x)
print(x[0])
print(x[1:3])
print(x[0, 1, 1])
print(x[0, 1:3, 0])

print("=============")

x = DLL.randn((3, 3))
print(x)
print(x.transpose())
print(x[0:1, :])
print(x.transpose()[:, 0:1])
