import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVR
from termcolor import colored

from Function import Feature_loading

from sklearn.ensemble import RandomForestRegressor

from termcolor import colored

from xgboost import XGBRegressor
import numpy as np
import matplotlib.pyplot as plt
from termcolor import colored


# Improved regression model function with XGBoost
def XGBoost_Regressor(xtrain, ytrain):
    print(colored("XGBoost Regression  ---->> ", color='blue', on_color='on_grey'))

    # Initialize XGBoost regressor
    model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)

    # Train the model
    model.fit(xtrain, ytrain.flatten())

    # Predict the next 7 days
    last_known = xtrain[-1].copy()  # Take the last known input
    future_predictions = []
    true_predictions_nxt_7days = [74014.55, 73903.91, 73876.82, 74227.63, 74248.22, 74742.50, 74683.70]  # Ground truth

    for i in range(7):  # Predict for the next 7 days
        next_day = model.predict(last_known.reshape(1, -1))  # Predict based on current input
        future_predictions.append(next_day[0])
        # Update input for next prediction
        last_known = np.roll(last_known, shift=-1)  # Shift input left
        last_known[-1] = next_day  # Add the new prediction at the end

    days = np.arange(1, 8)  # Days for the future predictions
    day_labels = [f"Day {i}" for i in days]
    plt.figure(figsize=(10, 6))
    plt.plot(days, future_predictions, label='Predicted', marker='o', linestyle='--', color='green')
    plt.plot(days, true_predictions_nxt_7days, label='True', marker='x', linestyle='-', color='blue')
    plt.title('True vs Predicted')
    plt.xlabel('Days')
    plt.ylabel('Values')
    plt.xticks(days, day_labels)
    plt.legend()
    plt.grid()
    plt.show()

    return model, future_predictions


# Load features and labels
feat, lab = Feature_loading('DS1')
feat = feat.astype(np.float32) / feat.max()  # Normalize features

# Train and predict using the improved XGBoost model
model, predicted = XGBoost_Regressor(feat, lab)




import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

def LSTM_Regressor(xtrain, ytrain):
    print(colored("LSTM Regression  ---->> ", color='blue', on_color='on_grey'))

    # Define the LSTM model
    model = Sequential([
        LSTM(100, activation='relu', return_sequences=True, input_shape=(xtrain.shape[1], xtrain.shape[2])),
        LSTM(50, activation='relu'),
        Dense(1)  # Output layer for regression
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='mse')

    # Train the model
    history = model.fit(xtrain, ytrain, epochs=100, batch_size=16, verbose=0)

    # Predict the next 7 days
    last_known = xtrain[-1].copy()  # Start with the last known time step
    future_predictions = []
    true_predictions_nxt_7days = [74014.55, 73903.91, 73876.82, 74227.63, 74248.22, 74742.50, 74683.70]  # Ground truth

    for i in range(7):
        next_day = model.predict(last_known[np.newaxis, :, :])  # Add batch dimension for prediction
        future_predictions.append(next_day[0][0])
        # Roll the sequence and append the new prediction
        last_known = np.roll(last_known, shift=-1, axis=0)
        last_known[-1, :] = next_day  # Replace the last time step with the new prediction

    # Plot true and predicted values
    days = np.arange(1, 8)
    plt.figure(figsize=(10, 6))
    plt.plot(days, future_predictions, label='Predicted', marker='o', linestyle='--', color='green')
    plt.plot(days, true_predictions_nxt_7days, label='True', marker='x', linestyle='-', color='blue')
    plt.title('Next 7 Days: True vs Predicted')
    plt.xlabel('Days')
    plt.ylabel('Values')
    plt.legend()
    plt.grid()
    plt.show()

    return model, future_predictions



# Load features and labels
feat, lab = Feature_loading('DS1')

# Preprocessing: Normalize and reshape for LSTM
feat = feat.astype(np.float32) / feat.max()  # Normalize features
lab = lab.astype(np.float32)
xtrain = np.expand_dims(feat, axis=1)  # Add time dimension for LSTM
ytrain = lab

# Train and predict using the LSTM model
model, predicted = LSTM_Regressor(xtrain, ytrain)


# Improved regression model function
def Random_Forest_Regressor(xtrain, ytrain):
    print(colored("Random Forest Regression  ---->> ", color='blue', on_color='on_grey'))

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    # Train the data
    model.fit(xtrain, ytrain.flatten())

    # Predict the next 7 days (example assumes sequential data)
    last_known = xtrain[-1].reshape(1, -1)
    future_predictions = []
    true_predictions_nxt_7days = [74014.55, 73903.91, 73876.82, 74227.63, 74248.22, 74742.50, 74683.70]  # Ground truth

    for i in range(7):  # Predict for the next 7 days
        next_day = model.predict(last_known)  # Predict based on current input
        future_predictions.append(next_day[0])
        # Update the input to include the new prediction (assuming sequential dependencies)
        last_known = np.append(last_known[:, 1:], [[next_day[0]]], axis=1)

    # Plot true and predicted values
    days = np.arange(1, 8)  # Days for the future predictions
    plt.figure(figsize=(10, 6))
    plt.plot(days, future_predictions, label='Predicted', marker='o', linestyle='--', color='green')
    plt.plot(days, true_predictions_nxt_7days, label='True', marker='x', linestyle='-', color='blue')
    plt.title('Next 7 Days: True vs Predicted')
    plt.xlabel('Days')
    plt.ylabel('Values')
    plt.legend()
    plt.grid()
    plt.show()

    return model, future_predictions


# Load features and labels
feat, lab = Feature_loading('DS1')
feat = feat.astype(np.float32) / feat.max()  # Normalize features

# Train and predict using the improved model
model, predicted = Random_Forest_Regressor(feat, lab)



















def Support_Vector_Regressor(xtrain, ytrain):
    print(colored("Support Vector Regression  ---->> ", color='blue', on_color='on_grey'))

    model = SVR(kernel='linear')
    # train the data
    model.fit(xtrain, ytrain.flatten())

    # Predict the next 7 days (example assumes sequential data)
    # Using the last available feature row for demonstration
    last_known = feat[-1].reshape(1, -1)
    future_predictions = []
    true_predictions_nxt_7days = [74014.55, 73903.91, 73876.82, 74227.63, 74248.22, 74742.50, 74683.70]
    for i in range(7):  # Predict for the next 7 days
        next_day = model.predict(last_known)  # Predict based on current input
        future_predictions.append(next_day[0])
        # Update the input to include the new prediction (assuming sequential dependencies)
        last_known = np.append(last_known[:, 1:], [[next_day[0]]], axis=1)

    # Plot the results
    days = np.arange(1, 8)  # Days for the future predictions
    plt.figure(figsize=(10, 6))
    plt.plot(days, future_predictions, label='Predicted', marker='o', linestyle='--', color='green')
    plt.title('Next 7 Days Prediction')
    plt.xlabel('Days')
    plt.ylabel('Predicted Values')
    plt.legend()
    plt.grid()
    plt.show()


feat, lab = Feature_loading('DS1')
f = feat.astype(np.float32) / feat.max()
ytrue1, pred1 = Support_Vector_Regressor(f, lab)






# ----------- for execution time plot -----------
def Execution_time_plot():
    # Example data
    models = ['Model 1', 'Model 2', 'Model 3', 'Model 4', 'Model 5', 'Model 6', 'Model 7']
    execution_times = [12.5, 10.8, 15.3, 9.7, 14.2, 11.4, 13.0]  # Replace with your data

    # Bar plot
    plt.figure(figsize=(10, 6))
    plt.bar(models, execution_times, color='green')

    # Add labels and title
    plt.xlabel('Models', fontsize=12)
    plt.ylabel('Execution Time (seconds)', fontsize=12)
    plt.title('Execution Time of Different Models', fontsize=14)

    # Show the values on the bars
    for i, v in enumerate(execution_times):
        plt.text(i, v + 0.3, str(v), ha='center', fontsize=10)

    # Display the plot
    plt.tight_layout()
    plt.show()