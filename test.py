import numpy as np

m = 3000
a = np.random.normal(size=m)
b = np.random.normal(size=m)
n = np.random.normal(size=m)

h2, h3 = 5, 3
s2 = 1 + h2**2 + h3**2
res = h2*a + h3*b + n

est2 = 1/2 * np.log(2*np.pi*s2) + 1/(2*m*s2) * np.linalg.norm(res,2)**2
print(est2)

print(1/2 + 1/2*np.log(2*np.pi) + 1/2*np.log(s2) )