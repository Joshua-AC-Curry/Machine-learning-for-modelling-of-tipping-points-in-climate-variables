import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.utils.vis_utils import plot_model
import ruptures as rpt
import os

#unit test to check if the predictions arrays exist
assert os.path.isfile('predictionsArray.npy') == True
assert os.path.isfile('indiPredictionsArray.npy') == True

predictions = np.load('predictionsArray.npy')
indipreds = np.load('indiPredictionsArray.npy')

# Set the font size
plt.rcParams['font.size'] = 18

option = 1
#unit tests to enusre option is the correct type
assert isinstance(option, int) == True

if option == 1:
    plt.title("Line graph for temperatures in the next 100 years")
    #plt.plot(range(1, 50 + 1), predictions, color="red")
    x = range(1, 100 + 1)
    a, b = np.polyfit(x, predictions, 1)
    plt.scatter(x, predictions)
    print("For next 100 years")
    #Drawing Line of best fit
    plt.plot(x, a*x+b) 
    plt.show()
elif option == 2:
    plt.title("Line graph for temperatures in the next 250 months")
    #plt.plot(range(1, 50 + 1), predictions, color="red")
    res = indipreds[:250]
    x = range(1, len(res) + 1)
    a, b = np.polyfit(x, res, 1)
    plt.plot(x, res)
    print("For next 250 days")
    #Drawing Line of best fit
    plt.plot(x, a*x+b) 
    plt.show()

mPlot = False
#unit tests to enusre mPlot is the correct type
assert isinstance(mPlot, bool) == True
if(mPlot):
    #unit test to ensure the model exists
    assert os.path.isfile('TempPredictions_TimeSeriesCV.keras') == True

    model = tf.keras.models.load_model('TempPredictions_TimeSeriesCV.keras')

    plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

tableValues = False
#unit tests to enusre tableValues is the correct type
assert isinstance(tableValues, bool) == True
if tableValues:
    np.savetxt("predictionValues.csv", indipreds, delimiter=",")

graphTippingPoints = False
#unit tests to enusre graphTippingPoints is the correct type
assert isinstance(graphTippingPoints, bool) == True
if graphTippingPoints:
    #code for grapinh preds with tipping points
    plt.title("Scatter graph for temperatures in the next 100 years")
    #plt.plot(range(1, 50 + 1), predictions, color="red")
    x = range(1, 100 + 1)
    a, b = np.polyfit(x, predictions, 1)
    plt.scatter(x, predictions)
    print("For next 50 years")
    #Drawing Line of best fit
    plt.plot(x, a*x+b) 
    plt.axhline(y=9.03125, color='r', linestyle='-')
    plt.axhline(y=9.53125, color='r', linestyle='-')
    plt.show()

