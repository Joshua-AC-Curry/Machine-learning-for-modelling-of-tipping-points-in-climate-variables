import numpy as np
import matplotlib.pyplot as plt


predictions = np.load('predictionsArray.npy')
indipreds = np.load('indiPredictionsArray.npy')

option = 2

if option == 1:
    plt.title("Line graph for temperatures in the next 50 years")
    #plt.plot(range(1, 50 + 1), predictions, color="red")
    x = range(1, 50 + 1)
    a, b = np.polyfit(x, predictions, 1)
    plt.scatter(x, predictions)
    print("For next 100 years")
    #Drawing Line of best fit
    plt.plot(x, a*x+b) 
    plt.show()
elif option == 2:
    plt.title("Line graph for temperatures in the next 50 years")
    #plt.plot(range(1, 50 + 1), predictions, color="red")
    x = range(1, len(indipreds) + 1)
    a, b = np.polyfit(x, indipreds, 1)
    plt.plot(x, indipreds)
    print("For next 100 years")
    #Drawing Line of best fit
    #plt.plot(x, a*x+b) 
    plt.show()