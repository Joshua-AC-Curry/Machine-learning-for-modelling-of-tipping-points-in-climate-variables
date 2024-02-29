import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Load the dataset
dataset_path = 'GlobalTemperatures.csv'
df = pd.read_csv(dataset_path)

print(df.head())

# Extract relevant columns
df = df[['dt', 'LandAverageTemperature']]
df['dt'] = pd.to_datetime(df['dt'])
df['dt'] = pd.to_numeric(df['dt'])
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
X_sequence, y_sequence = [], []

for i in range(len(X_scaled) - sequence_length):
    X_sequence.append(X_scaled[i:i + sequence_length])
    y_sequence.append(y_scaled[i + sequence_length])

X_sequence = np.array(X_sequence)
y_sequence = np.array(y_sequence)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_sequence, y_sequence, test_size=0.2, random_state=42)

# Build the LSTM model
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(64, activation='relu', input_shape=(sequence_length, 1), return_sequences=True, kernel_initializer='normal'),
    tf.keras.layers.LSTM(32, activation='relu', return_sequences=True, kernel_initializer='normal'),
    tf.keras.layers.LSTM(32, activation='relu', return_sequences=True, kernel_initializer='normal'),
    tf.keras.layers.LSTM(32, activation='relu', return_sequences=True, kernel_initializer='normal'),
    tf.keras.layers.LSTM(32, activation='relu', return_sequences=True, kernel_initializer='normal'),
    tf.keras.layers.LSTM(32, activation='relu', return_sequences=True, kernel_initializer='normal'),
    tf.keras.layers.LSTM(32, activation='relu', return_sequences=True, kernel_initializer='normal'),
    tf.keras.layers.LSTM(16, activation='relu', kernel_initializer='normal'),
    tf.keras.layers.Dense(1, activation='linear', kernel_initializer='normal')
]) #Note: too many layers overfitting so less layers means trend still goes up

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=16, validation_split=0.2)

# Evaluate the model on the test set
loss = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss:.4f}')

# Saving model
model.save('TempPredictions.keras')