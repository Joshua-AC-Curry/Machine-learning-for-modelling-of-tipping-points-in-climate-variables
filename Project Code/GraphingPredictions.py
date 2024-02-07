import numpy as np
import matplotlib.pyplot as plt


predictions = np.load('predictionsArray.npy')

plt.title("Line graph for temperatures in the next 30 years")
plt.plot(range(1, 30 + 1), predictions, color="red")
print("For next 30 years")
plt.show()