# difficult to combine cell cMDS and invalid cell plots to one
import matplotlib.pyplot as plt

# plt.figure(figsize=(8,8))
# plt.subplot(3,2,1)
# plt.subplot(3,2,3)
# plt.subplot(3,2,5)
# plt.subplot(2,2,2)
# plt.subplot(2,2,4)
plt.figure(figsize=(20,8))
plt.subplot(1,3,1)
plt.subplot(1,3,3)
plt.subplot(3,3,2, figsize=(2.5,2.5))
# plt.axes('equal')
plt.subplot(3,3,5, figsize=(2.5,2.5))
# plt.axes('equal')
plt.subplot(3,3,8, figsize=(2.5,2.5))
# plt.axes('equal')
plt.savefig("tmp.png")