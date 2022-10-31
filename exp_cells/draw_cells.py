import matplotlib.pyplot as plt
import numpy as np



index = 89
x = np.loadtxt("outputs/cells_tmp_"+str(index)+"_0.txt")
y = np.loadtxt("outputs/cells_tmp_"+str(index)+"_1.txt")

plt.scatter(x, y, color = "red")

plt.savefig("./outputs/cells_tmp_"+str(index)+".png")