import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

x = [4, 8, 16, 32, 64, 128, 256, 512, 1024] # Number of nodes
y = [ 0.005,0.014,0.065,0.378,2.265] # Sequential version times
y2 = [0.943,1.088, 1.841,5.055,17.113] # Basic parallel version times
y3 = [0.082,0.214,0.561,1.445,4.652] # Improved parallel version times

fig, ax = plt.subplots()
ax.plot(x, y, y2, y3)
plt.savefig("chart.jpg")