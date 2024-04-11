import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.utils.vis_utils import plot_model


predictions = np.load('predictionsArray.npy')
indipreds = np.load('indiPredictionsArray.npy')

option = 3

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
    plt.title("Line graph for temperatures in the next 250 days")
    #plt.plot(range(1, 50 + 1), predictions, color="red")
    res = indipreds[:250]
    x = range(1, len(res) + 1)
    a, b = np.polyfit(x, res, 1)
    plt.plot(x, res)
    print("For next 250 days")
    #Drawing Line of best fit
    plt.plot(x, a*x+b) 
    plt.show()

mPlot = True
if(mPlot):
    model = tf.keras.models.load_model('TempPredictions_TimeSeriesCV.keras')

    plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)