from DLL import randn, ones


print(randn((2, 3)))
print(randn((1, 2)))
print(randn((2, 1)))
print(randn((2, 3, 4)))
print(randn((2, 3, 4, 5)))

print(ones((2, 3)))
print(ones((2, 3)).sum(dim=0, keepdim=True))
print(ones((2, 3)).sum(dim=1, keepdim=True))
print(ones((2, 3)).sum(dim=0, keepdim=False))
print(ones((2, 3)).sum(dim=1, keepdim=False))
