import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import TensorBoard, EarlyStopping
from statsmodels.tsa.seasonal import seasonal_decompose

tf.random.set_seed(3)  # Set random seed for reproducibility


# Function to load all sheets from an Excel file into a dictionary with error handling
def load_data(file_path, sheets):
    data_dict = {}
    for key, sheet in sheets.items():
        try:
            data_dict[key] = pd.read_excel(file_path, sheet_name=sheet)
            print(f"Loaded data for {key} from sheet {sheet}")
        except Exception as e:
            print(f"Failed to load data for {key} from sheet {sheet}: {e}")
    return data_dict


# Define the file path and sheet names with their corresponding keys
file_path = "Energy consumption data.xlsx"
sheets = {
    "building_1_gas": "B1G",
    "building_1_electricity": "B1E",
    "building_2_gas": "B2G",
    "building_2_electricity": "B2E",
    "building_3_gas": "B3G",
    "building_3_electricity": "B3E",
}

# Load the data
dictionary = load_data(file_path, sheets)


# Function to visualize data
def visualize_data(df, title, xlabel, ylabel):
    sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})
    sns.set(rc={"figure.figsize": (15, 6)})
    sns.histplot(df["Daily consumption"], bins=50, kde=True, color="blue")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


# Function to create and compile the model
def create_model(input_shape, output_shape):
    model = Sequential()
    model.add(LSTM(64, input_shape=input_shape))
    model.add(Dense(16, activation="sigmoid"))
    model.add(Dense(output_shape))
    model.compile(
        optimizer="adam", loss="mean_squared_error", metrics=["mse", "mae", "mape"]
    )
    return model


# Function to plot data
def plot_the_data(orig_plot, predict_train_plot, predict_test_plot, title):
    plt.figure(figsize=(15, 6))
    plt.plot(orig_plot, color="blue", label="Actual")
    plt.plot(predict_train_plot, color="red", label="Predicted on training")
    plt.plot(predict_test_plot, color="green", label="Predicted on testing")
    plt.legend()
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Energy Consumption (kWh)")
    plt.show()


# Model inputs
train_size = 0.7  # 70% of the data is used for training
lookback = 7  # 7 days of data is used to predict the next day

# Example for building 1 (Electricity)
building_name = "building_1_electricity"
df = dictionary[building_name]
df["Daily consumption"] = df.iloc[:, 1:-3].sum(axis=1)

# Visualize data
visualize_data(df, "Energy Distribution (kWh)", "Daily Consumption (kWh)", "Frequency")


# Data preparation functions
def scale_split_datasets(data, train_size, lookback):
    sc_X = StandardScaler()
    daily_consumption_scaled = sc_X.fit_transform(data.values.reshape(-1, 1))
    num_train = int(train_size * len(data))
    training_data = daily_consumption_scaled[:num_train]
    test_data = daily_consumption_scaled[num_train - lookback :]
    return training_data, test_data, sc_X


def create_rnn_dataset(data, lookback):
    data_x, data_y = [], []
    for i in range(len(data) - lookback - 1):
        a = data[i : (i + lookback), 0]
        data_x.append(a)
        data_y.append(data[i + lookback, 0])
    x = np.array(data_x)
    y = np.array(data_y)
    x = np.reshape(x, (x.shape[0], 1, x.shape[1]))
    return x, y


def plot_data_preparation(data, predict_on_train, predict_on_test, lookback):
    total_size = len(predict_on_train) + len(predict_on_test)
    orig_data = data.to_numpy().reshape(len(data), 1)
    orig_plot = np.empty((total_size, 1))
    orig_plot[:, :] = np.nan
    orig_plot[:total_size, :] = orig_data[lookback:-2]
    predict_train_plot = np.empty((total_size, 1))
    predict_train_plot[:, :] = np.nan
    predict_train_plot[: len(predict_on_train), :] = predict_on_train
    predict_test_plot = np.empty((total_size, 1))
    predict_test_plot[:, :] = np.nan
    predict_test_plot[len(predict_on_train) : total_size, :] = predict_on_test
    return orig_plot, predict_train_plot, predict_test_plot


# Scale and split data
training_set, test_data, sc_X = scale_split_datasets(
    df["Daily consumption"], train_size, lookback
)
x_train, y_train = create_rnn_dataset(training_set, lookback)
x_test, y_test = create_rnn_dataset(test_data, lookback)

# Create and train the model
ts_model = create_model(input_shape=(1, lookback), output_shape=1)
log_dir = "logs/fit/" + dt.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

ts_model.fit(
    x_train,
    y_train,
    epochs=5,
    batch_size=1,
    verbose=1,
    callbacks=[tensorboard_callback],
)

# Evaluate model
ts_model.evaluate(x_test, y_test, verbose=1)
predict_on_train = ts_model.predict(x_train)
predict_on_test = ts_model.predict(x_test)

predict_on_train = sc_X.inverse_transform(predict_on_train)
predict_on_test = sc_X.inverse_transform(predict_on_test)

# Plot predictions
plot_original, plot_train, plot_test = plot_data_preparation(
    df["Daily consumption"], predict_on_train, predict_on_test, lookback
)
plot_the_data(plot_original, plot_train, plot_test, "Model Predictions vs Actual")

# Seasonal decomposition
result = seasonal_decompose(df["Daily consumption"], model="additive", period=365)
result.plot()
plt.suptitle("Seasonal Decomposition of Daily Consumption")
plt.show()

# Enhanced visualization of seasonal decomposition
fig, axes = plt.subplots(4, 1, figsize=(15, 12), sharex=True)
result.observed.plot(ax=axes[0], title="Observed")
result.trend.plot(ax=axes[1], title="Trend")
result.seasonal.plot(ax=axes[2], title="Seasonal")
result.resid.plot(ax=axes[3], title="Residual")
for ax in axes:
    ax.set_ylabel("Energy Consumption (kWh)")
axes[3].set_xlabel("Time")
plt.tight_layout()
plt.show()

# Save the residuals to an Excel file
os.makedirs(building_name, exist_ok=True)
pd.DataFrame(result.resid).to_excel(
    f"{building_name}/building.xlsx", sheet_name="Decomposition_Residuals"
)

# Summary statistics
residual_stats = pd.DataFrame(
    {
        "Mean": [result.resid.mean()],
        "Median": [result.resid.median()],
        "Standard Deviation": [result.resid.std()],
        "Max": [result.resid.max()],
        "Min": [result.resid.min()],
    }
)

# Save summary statistics
residual_stats.to_excel(f"{building_name}/residual_statistics.xlsx", index=False)

# Display summary statistics
print("Summary Statistics of Residuals:")
print(residual_stats)

# Evaluation metrics
metrics = ts_model.evaluate(x_test, y_test, verbose=1)
metrics_df = pd.DataFrame([metrics], columns=["Loss", "MSE", "MAE", "MAPE"])

# Save evaluation metrics
metrics_df.to_excel(f"{building_name}/evaluation_metrics.xlsx", index=False)

# Display evaluation metrics
print("Evaluation Metrics:")
print(metrics_df)


# Federated Learning Model Preparation
def create_train_test_dataset(df, lookback):
    df["Daily consumption"] = df.iloc[:, 1:-3].sum(axis=1)
    sc_X = StandardScaler()
    daily_consumption = df["Daily consumption"]
    num_train = int(train_size * len(daily_consumption))
    daily_consumption_scaled = sc_X.fit_transform(
        daily_consumption.values.reshape(-1, 1)
    )
    training_set = daily_consumption_scaled[:num_train]
    x_train, y_train = create_rnn_dataset(training_set, lookback)
    test_data = daily_consumption_scaled[num_train - lookback :]
    x_test, y_test = create_rnn_dataset(test_data, lookback)
    return x_train, y_train, x_test, y_test, sc_X


b1e_xtrain, b1e_ytrain, b1e_xtest, b1e_ytest, sc_b1e = create_train_test_dataset(
    dictionary["building_1_electricity"], lookback
)
b2e_xtrain, b2e_ytrain, b2e_xtest, b2e_ytest, sc_b2e = create_train_test_dataset(
    dictionary["building_2_electricity"], lookback
)
b3e_xtrain, b3e_ytrain, b3e_xtest, b3e_ytest, sc_b3e = create_train_test_dataset(
    dictionary["building_3_electricity"], lookback
)

b1e_model = create_model(input_shape=(1, lookback), output_shape=1)
b2e_model = create_model(input_shape=(1, lookback), output_shape=1)
b3e_model = create_model(input_shape=(1, lookback), output_shape=1)


def train_model(model, x_train, y_train, log_dir):
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
    early_stopping = EarlyStopping(
        monitor="loss", min_delta=0.001, patience=5, verbose=1, mode="auto"
    )
    model.fit(
        x_train,
        y_train,
        epochs=60,
        batch_size=1,
        verbose=1,
        callbacks=[tensorboard_callback, early_stopping],
    )


train_model(b3e_model, b3e_xtrain, b3e_ytrain, "logs/fit/B3/")
train_model(b1e_model, b1e_xtrain, b1e_ytrain, "logs/fit/B1/")
train_model(b2e_model, b2e_xtrain, b2e_ytrain, "logs/fit/B2/")


# Evaluate the models
def evaluate_model(model, x_train, y_train, x_test, y_test):
    model.evaluate(x_train, y_train)
    model.evaluate(x_test, y_test)


evaluate_model(b3e_model, b3e_xtrain, b3e_ytrain, b3e_xtest, b3e_ytest)
evaluate_model(b1e_model, b1e_xtrain, b1e_ytrain, b1e_xtest, b1e_ytest)
evaluate_model(b2e_model, b2e_xtrain, b2e_ytrain, b2e_xtest, b2e_ytest)


# Federated model training
def federated_averaging(models, x_train, y_train, rounds=10):
    for _ in range(rounds):
        weights = [model.get_weights() for model in models]
        new_weights = [
            np.mean([w[i] for w in weights], axis=0) for i in range(len(weights[0]))
        ]
        for model in models:
            model.set_weights(new_weights)
            model.fit(x_train, y_train, epochs=5, batch_size=1, verbose=1)


models = [b1e_model, b2e_model, b3e_model]
federated_averaging(models, b1e_xtrain, b1e_ytrain)
federated_averaging(models, b2e_xtrain, b2e_ytrain)
federated_averaging(models, b3e_xtrain, b3e_ytrain)


# Predictions and plotting
def inverse_transform_predictions(predictions, scaler):
    return scaler.inverse_transform(predictions)


b1_train_predictions = inverse_transform_predictions(
    b1e_model.predict(b1e_xtrain), sc_b1e
)
b2_train_predictions = inverse_transform_predictions(
    b2e_model.predict(b2e_xtrain), sc_b2e
)
b3_train_predictions = inverse_transform_predictions(
    b3e_model.predict(b3e_xtrain), sc_b3e
)

b1_test_predictions = inverse_transform_predictions(
    b1e_model.predict(b1e_xtest), sc_b1e
)
b2_test_predictions = inverse_transform_predictions(
    b2e_model.predict(b2e_xtest), sc_b2e
)
b3_test_predictions = inverse_transform_predictions(
    b3e_model.predict(b3e_xtest), sc_b3e
)


def prepare_and_plot(data, train_predictions, test_predictions, lookback, title):
    orig_plot, train_plot, test_plot = plot_data_preparation(
        data, train_predictions, test_predictions, lookback
    )
    plot_the_data(orig_plot, train_plot, test_plot, title)


prepare_and_plot(
    dictionary["building_1_electricity"]["Daily consumption"],
    b1_train_predictions,
    b1_test_predictions,
    lookback,
    "Federated Model - Building 1",
)
prepare_and_plot(
    dictionary["building_2_electricity"]["Daily consumption"],
    b2_train_predictions,
    b2_test_predictions,
    lookback,
    "Federated Model - Building 2",
)
prepare_and_plot(
    dictionary["building_3_electricity"]["Daily consumption"],
    b3_train_predictions,
    b3_test_predictions,
    lookback,
    "Federated Model - Building 3",
)

# Save the predictions
output_dir = "federated"
os.makedirs(output_dir, exist_ok=True)

predictions_files = {
    "b1_test_predictions": b1_test_predictions,
    "b2_test_predictions": b2_test_predictions,
    "b3_test_predictions": b3_test_predictions,
    "b1_train_predictions": b1_train_predictions,
    "b2_train_predictions": b2_train_predictions,
    "b3_train_predictions": b3_train_predictions,
    "b1_original": dictionary["building_1_electricity"]["Daily consumption"],
    "b2_original": dictionary["building_2_electricity"]["Daily consumption"],
    "b3_original": dictionary["building_3_electricity"]["Daily consumption"],
}

for filename, data in predictions_files.items():
    pd.DataFrame(data).to_excel(f"{output_dir}/{filename}.xlsx")
