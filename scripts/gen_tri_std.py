import sys
n = int(sys.argv[1])
print("MFEM mesh v1.0")
print("dimension")
print("2")
print("elements")
print(2 * n * n)
for j in range(n):
    for i in range(n):
        a = j * (n + 1) + i + 1
        b = a + 1
        c = (j + 1) * (n + 1) + i + 2
        d = c - 1
        print(f"1 2 {a} {b} {c}")
        print(f"1 2 {a} {c} {d}")
print("boundary")
print(4 * n)
for i in range(n):
    print(f"1 1 {i+1} {i+2}")
for i in range(n):
    print(f"1 1 {(i+1)*(n+1)} {(i+2)*(n+1)}")
for i in range(n):
    print(f"1 1 {n*(n+1)+i+1} {n*(n+1)+i+2}")
for i in range(n):
    print(f"1 1 {i*(n+1)+1} {(i+1)*(n+1)+1}")
print("vertices")
print((n + 1) * (n + 1))
print("2")
for j in range(n + 1):
    for i in range(n + 1):
        print(f"{i/n} {j/n}")
