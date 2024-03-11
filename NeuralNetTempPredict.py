import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler

# Load the dataset
dataset_path = 'GlobalTemperatures.csv'
df = pd.read_csv(dataset_path)

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
sequence_length = 10
X_sequence, y_sequence = [], []

for i in range(len(X_scaled) - sequence_length):
    X_sequence.append(X_scaled[i:i + sequence_length])
    y_sequence.append(y_scaled[i + sequence_length])

X_sequence = np.array(X_sequence)
y_sequence = np.array(y_sequence)

# Define the number of splits for time series cross-validation
num_splits = 5
# Initialize TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=num_splits)

# Initialize a list to store the model's performance metrics for each fold
test_losses = []

# Loop through each fold
for train_index, test_index in tscv.split(X_sequence):
    # Split data into train and test sets for this fold
    X_train, X_test = X_sequence[train_index], X_sequence[test_index]
    y_train, y_test = y_sequence[train_index], y_sequence[test_index]
    
    # Build the LSTM model
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(64, activation='relu', input_shape=(sequence_length, 1), return_sequences=True, kernel_initializer='normal'),
        tf.keras.layers.LSTM(32, activation='relu', return_sequences=True, kernel_initializer='normal'),
        tf.keras.layers.LSTM(16, activation='relu', kernel_initializer='normal'),
        tf.keras.layers.Dense(1, activation='linear', kernel_initializer='normal')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(X_train, y_train, epochs=100, batch_size=16, verbose=0)
    
    # Evaluate the model on the test set for this fold
    loss = model.evaluate(X_test, y_test)
    test_losses.append(loss)

# Calculate the mean test loss across all folds
mean_test_loss = sum(test_losses) / len(test_losses)
print(f'Mean Test Loss: {mean_test_loss:.4f}')

# Saving model
model.save('TempPredictions_TimeSeriesCV.keras')
