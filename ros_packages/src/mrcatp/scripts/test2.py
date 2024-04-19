import numpy as np

def updateArray(p):
    p[:] = np.random.uniform(low=-1,high=1,size=p.shape)


p = np.zeros(shape=(2,5))
updateArray(p)
print(p)
