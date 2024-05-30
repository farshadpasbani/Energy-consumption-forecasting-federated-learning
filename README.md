# Purpose
The purpose of this code is to load, preprocess, analyze, and model energy consumption data for different buildings. The ultimate goal is to predict future energy consumption using a Recurrent Neural Network (RNN) model (specifically an LSTM model), visualize the predictions, and perform federated learning to aggregate and improve model performance across multiple buildings.

## Installation
### Prerequisites
Ensure you have the following installed on your system:

- Python 3.6 or higher
- pip (Python package installer)

### Required Libraries
The required Python libraries can be installed using the following command:
```sh
pip install -r requirements.txt
```
Alternatively, you can run the program if you have the following packages installed:
```
numpy
pandas
matplotlib
seaborn
tensorflow
keras
statsmodels
openpyxl
```

### Download the Repository
Clone the repository to your local machine:
```sh
git clone https://github.com/farshadpasbani/Energy-consumption-forecasting-federated-learning.git
```

## Usage
### Step 1: Load the Data
Ensure your half-hourly energy consumption data is in an Excel file with sheets corresponding to different buildings and energy types. Update the `file_path` and `sheets` dictionary in the script to match your data.

### Step 2: Run the Script
Execute the main script to load the data, train the model, and generate predictions:
```sh
python main.py
```
### Step 3: View the Results
After running the script, the following outputs will be generated:
**Visualizations**: Histograms of daily energy consumption and comparison plots of actual vs. predicted values.
**Seasonal Decomposition**: Plots showing the observed, trend, seasonal, and residual components of the time series data.
**Summary Statistics**: Excel files containing summary statistics of the residuals.
**Model Predictions**: Excel files containing the model's predictions for training and test sets.
**Federated Learning Results**: Combined model predictions using federated learning across multiple buildings.

## Project Structure
`main.py`: Main script for loading data, training the model, and generating predictions.
`requirements.txt`: List of required Python libraries.
`README.md`: This file.

# Detailed Explanation and Steps:
## 1. Load Data:
- The code begins by defining a function `load_data` to read multiple sheets from an Excel file into a dictionary. Each sheet corresponds to energy consumption data for a specific building and energy type (e.g., gas or electricity).
- This is done using `pd.read_excel`, and the data is stored in a dictionary with keys representing each building and energy type.

## 2. Visualization:
- The function `visualize_data` is used to create histograms of daily energy consumption data. This provides an initial view of the data distribution.
- Seaborn's `histplot` function is used for visualization, with titles and axis labels added to make the plots informative.

## 3. Data Preparation:
- The data is preprocessed to make it suitable for training the LSTM model. This includes:
  - Scaling the data using `StandardScaler` to normalize the energy consumption values.
  - Splitting the data into training and test sets based on a specified `train_size`.
  - Creating RNN-compatible datasets using the function `create_rnn_dataset`, which generates sequences of data points for the model to learn from.

## 4. Model Creation and Training:
- The function `create_model` defines the architecture of the LSTM model. The model consists of one LSTM layer, followed by two Dense layers.
- The model is compiled with the Adam optimizer and the mean squared error loss function.
- The model is trained on the training data using TensorFlow's `fit` method, with TensorBoard callbacks for monitoring training progress.

## 5. Model Evaluation and Prediction:
- After training, the model's performance is evaluated on the test set to ensure it generalizes well to unseen data.
- Predictions are made for both training and test sets, and these predictions are inverse-transformed to return them to the original scale.
- The function `plot_the_data` is used to plot the actual versus predicted energy consumption values, providing a visual comparison of model performance.

## 6. Seasonal Decomposition:
- Seasonal decomposition is performed using the `seasonal_decompose` function from the statsmodels library. This decomposes the time series data into its seasonal, trend, and residual components.
- The decomposition is visualized using matplotlib, with clear titles and axis labels.

## 7. Summary Statistics:
- Summary statistics of the residuals from the seasonal decomposition are calculated and saved to an Excel file. These statistics include the mean, median, standard deviation, max, and min of the residuals.
- This provides insights into the residuals' distribution and can help identify any patterns or anomalies.

## 8. Federated Learning:
- Federated learning is implemented to train models on data from multiple buildings and aggregate their learning. This is achieved using the `federated_averaging` function.
- Separate models are trained for each building, and their weights are averaged and updated iteratively. This approach helps improve model performance by leveraging data from multiple sources without centralizing the data.
- The predictions from the federated model are plotted and compared to the actual energy consumption values.

## 9. Saving Predictions:
- The predictions from the models are saved to Excel files for further analysis or reporting. This includes predictions for both training and test sets, as well as the original daily consumption data.
