import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import os

#unit test to make sure the files exists
assert os.path.isfile('TempPredictions_TimeSeriesCV.keras') == True
model = tf.keras.models.load_model('TempPredictions_TimeSeriesCV.keras')

model.summary()

# Load the dataset
# Replace 'path_to_dataset.csv' with the actual path to your downloaded CSV file
dataset_path = 'GlobalTemperatures.csv'
#unit test to make sure the files exists
assert os.path.isfile(dataset_path) == True
df = pd.read_csv(dataset_path)

# Extract relevant columns
df = df[['dt', 'LandAverageTemperature']]
df['dt'] = pd.to_datetime(df['dt'])
df = df.set_index('dt')

# Add a column for temperature change
df['TemperatureChange'] = df['LandAverageTemperature'].diff()

# Drop the first row with NaN in TemperatureChange
df = df.dropna()

# Use the land average temperature as the target variable
X = df[['LandAverageTemperature']].values
y = df['LandAverageTemperature'].values

# Normalize the data using Min-Max scaling
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

# Create sequences for the RNN
sequence_length = 10  # You can adjust this based on the length of sequences you want to consider



# Make predictions on new data for the next 10 steps
year_predictions = []
new_temperature_data = X_scaled[-sequence_length:].reshape(1, sequence_length, 1)

year_predictions = []
number_of_years = 100
individual_predictions = []
for _ in range(number_of_years):
    predictions = []

    for _ in range(12):
        predicted_temperature_scaled = model.predict(new_temperature_data)[0, 0]
        prediction = scaler_y.inverse_transform([[predicted_temperature_scaled]])[0, 0]
        predictions.append(prediction)
        individual_predictions.append(prediction)

        # Update the input sequence for the next prediction
        new_temperature_data = np.roll(new_temperature_data, -1, axis=1)
        new_temperature_data[0, -1, 0] = predicted_temperature_scaled
    
    year_average = sum(predictions)/len(predictions)
    year_predictions.append(year_average)

#print(f'Predicted Temperatures for the Next 73000 Steps: {predictions}')

# Saving predictions to text file
year_predictions_np = np.array(year_predictions)
np.save('predictionsArray.npy', year_predictions_np)

individual_predictions_np = np.array(individual_predictions)
np.save('indiPredictionsArray.npy', individual_predictions_np)

#plt.title("Line graph for temperatures in the next 30 years")
#plt.plot(range(1, number_of_years + 1), year_predictions, color="red")
#print("For next 30 years")
#plt.show()