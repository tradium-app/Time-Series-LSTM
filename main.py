# %% Import Libraries
from core.model import Model
import os
import time
import math
from core.data_processor import DataLoader
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt

plt.style.use('./plot.mplstyle')

# %% Load data

df = pd.read_csv('./data/sp500.csv')

configs = json.load(open('model_config.json', 'r'))

data = DataLoader(
    os.path.join('data', configs['data']['filename']),
    configs['data']['train_test_split'],
    configs['data']['columns']
)


def plot_results_multiple(predicted_data, true_data, prediction_len):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    # Pad the list of predictions to shift it in the graph to it's correct start
    for i, data in enumerate(predicted_data):
        padding = [None for p in range(i * prediction_len)]
        plt.plot(padding + data, label='Prediction')
        plt.legend()
    plt.show()


# %% Build/Train the model
model = Model()

model.build_model(configs)

# out-of memory generative training
steps_per_epoch = math.ceil(
    (data.len_train - configs['data']['sequence_length']) / configs['training']['batch_size'])

model.train_generator(
    data_gen=data.generate_train_batch(
        seq_len=configs['data']['sequence_length'],
        batch_size=configs['training']['batch_size'],
        normalise=configs['data']['normalise']
    ),
    epochs=configs['training']['epochs'],
    batch_size=configs['training']['batch_size'],
    steps_per_epoch=steps_per_epoch,
    save_dir=configs['model']['save_dir']
)

x_test, y_test = data.get_test_data(
    seq_len=configs['data']['sequence_length'],
    normalise=configs['data']['normalise']
)

predictions = model.predict_sequences_multiple(
    x_test, configs['data']['sequence_length'], configs['data']['sequence_length'])

plot_results_multiple(predictions, y_test,
                      configs['data']['sequence_length'])
