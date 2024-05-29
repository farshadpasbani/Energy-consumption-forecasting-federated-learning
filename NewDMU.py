import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
import os

# Load the data
data_files = {
    "QG": ("New_DMU.xlsx", "QUENGF01"),
    "QE": ("New_DMU.xlsx", "QUENEF"),
    "HAG": ("New_DMU.xlsx", "HUASGF01"),
    "HAE": ("New_DMU.xlsx", "HUASEM"),
    "HTG": ("New_DMU.xlsx", "HAWTGF01"),
    "HTE": ("New_DMU.xlsx", "HAWTEMV"),
}

dictionary = {
    key: pd.read_excel(file, sheet_name=sheet)
    for key, (file, sheet) in data_files.items()
}

# Model inputs
train_size = 0.7  # 70% of the data is used for training
lookback = 7  # 7 days of data is used to predict the next day

# Individual model building for Hugh Electric
building_name = "HAE"
df = dictionary[building_name]
df["Daily consumption"] = df.iloc[:, 1:-3].sum(axis=1)

# Visualizing daily consumption
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})
sns.set(rc={"figure.figsize": (15, 6)})
sns.histplot(df["Daily consumption"], bins=50, kde=True, color="blue")
plt.title("Energy Distribution (kWh)")
plt.show()

# Import necessary libraries for model building
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import TensorBoard, EarlyStopping

tf.random.set_seed(3)  # Set random seed for reproducibility


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


def create_model(input_shape, output_shape):
    model = Sequential()
    model.add(LSTM(64, input_shape=input_shape))
    model.add(Dense(16, activation="sigmoid"))
    model.add(Dense(output_shape))
    model.compile(
        optimizer="adam", loss="mean_squared_error", metrics=["mse", "mae", "mape"]
    )
    return model


def plot_data_preparation(data, predict_on_train, predict_on_test, lookback):
    total_size = len(predict_on_train) + len(predict_on_test)
    orig_data = data.to_numpy().reshape(len(data), 1)

    orig_plot = np.empty((total_size, 1))
    orig_plot[:, :] = np.nan
    orig_plot[:total_size, :] = orig_data[lookback:-2,]

    predict_train_plot = np.empty((total_size, 1))
    predict_train_plot[:, :] = np.nan
    predict_train_plot[: len(predict_on_train), :] = predict_on_train

    predict_test_plot = np.empty((total_size, 1))
    predict_test_plot[:, :] = np.nan
    predict_test_plot[len(predict_on_train) : total_size, :] = predict_on_test

    return orig_plot, predict_train_plot, predict_test_plot


def plot_the_data(orig_plot, predict_train_plot, predict_test_plot):
    plt.plot(orig_plot, color="blue", label="Actual")
    plt.plot(predict_train_plot, color="red", label="Predicted on training")
    plt.plot(predict_test_plot, color="green", label="Predicted on testing")
    plt.legend()
    plt.show()


# Model training and evaluation
daily_consumption = df["Daily consumption"]
training_set, test_data, sc_X = scale_split_datasets(
    daily_consumption, train_size, lookback
)
x_train, y_train = create_rnn_dataset(training_set, lookback)
x_test, y_test = create_rnn_dataset(test_data, lookback)

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

ts_model.evaluate(x_test, y_test, verbose=1)
predict_on_train = ts_model.predict(x_train)
predict_on_test = ts_model.predict(x_test)

predict_on_train = sc_X.inverse_transform(predict_on_train)
predict_on_test = sc_X.inverse_transform(predict_on_test)

plot_original, plot_train, plot_test = plot_data_preparation(
    daily_consumption, predict_on_train, predict_on_test, lookback
)
plot_the_data(plot_original, plot_train, plot_test)

pd.DataFrame(plot_original).to_excel("plot_data.xlsx", index=False)
pd.DataFrame(plot_train).to_excel("plot_train.xlsx")
pd.DataFrame(plot_test).to_excel("plot_test.xlsx")

# Seasonal decomposition of the data
from statsmodels.tsa.seasonal import seasonal_decompose

tmpvar = float("inf")
residual_max = pd.DataFrame()

for best_period in range(3, 365):
    result = seasonal_decompose(plot_original, model="additive", period=best_period)
    residual_dataframe = pd.DataFrame(result.resid).dropna()
    compareval = residual_dataframe.max().values[0]
    residual_max = pd.concat([residual_max, residual_dataframe.max()])

    if compareval < tmpvar:
        tmpvar = compareval
        print(best_period, compareval)

residual_max.reset_index(drop=True, inplace=True)
residual_max.plot()
plt.show()
# Ensure the directory exists
os.makedirs(building_name, exist_ok=True)

# Save the residuals to an Excel file
pd.DataFrame(residual_max).to_excel(
    f"{building_name}/building.xlsx", sheet_name="Decomposition_Residuals"
)

result = seasonal_decompose(plot_original, model="additive", period=best_period)
result.plot()
plt.show()


# Federated learning model creation and training
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


queens_xtrain, queens_ytrain, queens_xtest, queens_ytest, sc_queens = (
    create_train_test_dataset(dictionary["QE"], lookback)
)
hugh_xtrain, hugh_ytrain, hugh_xtest, hugh_ytest, sc_hugh = create_train_test_dataset(
    dictionary["HAE"], lookback
)
hawthorn_xtrain, hawthorn_ytrain, hawthorn_xtest, hawthorn_ytest, sc_hawthorn = (
    create_train_test_dataset(dictionary["HTE"], lookback)
)

queens_model = create_model(input_shape=(1, lookback), output_shape=1)
hawthorn_model = create_model(input_shape=(1, lookback), output_shape=1)
hugh_model = create_model(input_shape=(1, lookback), output_shape=1)


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


train_model(hawthorn_model, hawthorn_xtrain, hawthorn_ytrain, "logs/fit/hawthorn/")
train_model(queens_model, queens_xtrain, queens_ytrain, "logs/fit/queens/")
train_model(hugh_model, hugh_xtrain, hugh_ytrain, "logs/fit/hugh/")


# Evaluate the models
def evaluate_model(model, x_train, y_train, x_test, y_test):
    model.evaluate(x_train, y_train)
    model.evaluate(x_test, y_test)


evaluate_model(
    hawthorn_model, hawthorn_xtrain, hawthorn_ytrain, hawthorn_xtest, hawthorn_ytest
)
evaluate_model(queens_model, queens_xtrain, queens_ytrain, queens_xtest, queens_ytest)
evaluate_model(hugh_model, hugh_xtrain, hugh_ytrain, hugh_xtest, hugh_ytest)


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


models = [queens_model, hugh_model, hawthorn_model]
federated_averaging(models, queens_xtrain, queens_ytrain)
federated_averaging(models, hugh_xtrain, hugh_ytrain)
federated_averaging(models, hawthorn_xtrain, hawthorn_ytrain)


# Predictions and plotting
def inverse_transform_predictions(predictions, scaler):
    return scaler.inverse_transform(predictions)


queens_train_predictions = inverse_transform_predictions(
    queens_model.predict(queens_xtrain), sc_queens
)
hugh_train_predictions = inverse_transform_predictions(
    hugh_model.predict(hugh_xtrain), sc_hugh
)
hawthorn_train_predictions = inverse_transform_predictions(
    hawthorn_model.predict(hawthorn_xtrain), sc_hawthorn
)

queens_test_predictions = inverse_transform_predictions(
    queens_model.predict(queens_xtest), sc_queens
)
hugh_test_predictions = inverse_transform_predictions(
    hugh_model.predict(hugh_xtest), sc_hugh
)
hawthorn_test_predictions = inverse_transform_predictions(
    hawthorn_model.predict(hawthorn_xtest), sc_hawthorn
)


def prepare_and_plot(data, train_predictions, test_predictions, lookback):
    orig_plot, train_plot, test_plot = plot_data_preparation(
        data, train_predictions, test_predictions, lookback
    )
    plot_the_data(orig_plot, train_plot, test_plot)


prepare_and_plot(
    dictionary["QE"]["Daily consumption"],
    queens_train_predictions,
    queens_test_predictions,
    lookback,
)
prepare_and_plot(
    dictionary["HAE"]["Daily consumption"],
    hugh_train_predictions,
    hugh_test_predictions,
    lookback,
)
prepare_and_plot(
    dictionary["HTE"]["Daily consumption"],
    hawthorn_train_predictions,
    hawthorn_test_predictions,
    lookback,
)

# Save the predictions
output_dir = "federated"
os.makedirs(output_dir, exist_ok=True)

predictions_files = {
    "queens_test_predictions": queens_test_predictions,
    "hugh_test_predictions": hugh_test_predictions,
    "hawthorn_test_predictions": hawthorn_test_predictions,
    "queens_train_predictions": queens_train_predictions,
    "hugh_train_predictions": hugh_train_predictions,
    "hawthorn_train_predictions": hawthorn_train_predictions,
    "queens_original": dictionary["QE"]["Daily consumption"],
    "hugh_original": dictionary["HAE"]["Daily consumption"],
    "hawthorn_original": dictionary["HTE"]["Daily consumption"],
}

for filename, data in predictions_files.items():
    pd.DataFrame(data).to_excel(f"{output_dir}/{filename}.xlsx")
