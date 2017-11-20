#!/usr/bin/env python3
import numpy as np
import numpy.linalg as LA


V1 = np.array([[1],[0],[0]])
V2 = np.array([[0],[1],[1]])

print("V1 orthogonal V2")
print()
print("V1: \n", V1)
print("V1 norm: \n", LA.norm(V1))
print()

print("V2:\n", V2)
print("V2 norm:\n", LA.norm(V2))
print()

print("V2 + V1:\n", V2+V1)
print("norm(V2 + V1) :\n", LA.norm(V2+V1))
print("norm(V2) + norm(V1) :\n", LA.norm(V2)+LA.norm(V1))

print("sqrt(norm(V2)^2 + norm(V1)^2):\n", (LA.norm(V2)**2+LA.norm(V1)**2)**(0.5))




