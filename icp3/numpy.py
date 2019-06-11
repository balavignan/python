import numpy as np
x = np.random.choice(range(1,21), size=15,replace=False)
print("Original Array:")
print(x)
x[x.argmax()] = 0
print("Maximum value replaced by 0")
print("replaced matrix:")
print(x)