import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

x = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096] # Number of nodes
x2 = [4, 8, 16, 32, 64, 128, 256, 512, 1024] # Number of nodes

#y = [0.0066, 0.0154, 0.2866, 0.4263, 2.1679, 14.6553, 116.1749, 922.7341, 7365.8792, 60335.5084, 493362.0435] # Sequential version times
#y2 = [0.5944, 0.4786, 1.0106, 4.4829, 16.0679, 60.5466, 200.3708, 732.9258, 3379.9178, 27177.7647, 335557.7241] # Basic parallel version times
#y3 = [0.0879, 0.2344, 0.6153, 1.6028, 4.4910, 28.5552, 242.2871, 2584.5715, 45256.2854] # Improved parallel version times

sequential = [0.0000066, 0.0000154, 0.0002866, 0.0004263, 0.0021679, 0.0146553, 0.1161749, 0.9227341, 7.3658792, 60.3355084, 493.3620435] # Sequential version times
basicParallel = [0.0005944, 0.0004786, 0.0010106, 0.0044829, 0.0160679, 0.0605466, 0.2003708, 0.7329258, 3.3799178, 27.1777647, 335.5577241] # Basic parallel version times
improvedParallel = [0.0000879, 0.0002344, 0.0006153, 0.0016028, 0.0044910, 0.0285552, 0.2422871, 2.5845715, 45.2562854] # Improved parallel version times

#for x in range(9):
#  print(sequential[x] / improvedParallel[x])

fig, ax = plt.subplots()
ax.plot(x, sequential, label="Sequential", marker='o')
ax.plot(x, basicParallel, label="Source Partitioned", marker='o')
ax.plot(x2, improvedParallel, label="Source Parallel", marker='o')
ax.set_title("Executions times for every algorithm")
ax.set_xlabel("Number of nodes")
ax.set_ylabel("Time in seconds")
ax.legend()
plt.savefig("chart.jpg")