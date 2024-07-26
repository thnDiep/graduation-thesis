import numpy as np
import random
import os
import pandas_ta as ta
import torch
import sys
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
from enum import Enum
from Action import Action
from Evaluation import Evaluation
import matplotlib.pyplot as plt


class StateMode(Enum):
    OHLC = 1
    WINDOWED = 2
    PI = 3
    CI = 4
    TI = 5
    TI_SIGNAL = 6
    SR_PERCENT_COL = 7
    SR_SIGNAL_COL = 8
    SR_PERCENT_ROW = 9
    SR_SIGNAL_ROW = 10
    MULTI_TIME_FRAME_SR_PERCENT_ROW = 11
    MULTI_TIME_FRAME_SR_SIGNAL_ROW = 12


def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def add_technical_indicators(data):
    CustomStrategy = ta.Strategy(
        name="Leading and lagging Indicators",
        description="Leading and Lagging Indicators",
        ta=[
            # LEADING
            {"kind": "rsi", "length": 14, "col_names": ("rsi")},
            {"kind": "stoch", "length": 14, "col_names": ("stochk", "stochd")},
            {"kind": "cci", "length": 14, "col_names": ("cci")},
            {"kind": "willr", "length": 14, "col_names": ("willr")},
            {"kind": "mom", "length": 10, "col_names": ("mom")},
            {"kind": "roc", "length": 10, "col_names": ("roc")},
            {"kind": "trix", "length": 5, "col_names": ("trix", "trixs")},
            {"kind": "cmo", "length": 5, "col_names": ("cmo")},

            # LAGGING
            {"kind": "sma", "length": 50, "col_names": ("sma_50")},
            {"kind": "sma", "length": 200, "col_names": ("sma_200")},
            {"kind": "ema", "length": 10, "col_names": ("ema_10")},
            {"kind": "ema", "length": 20, "col_names": ("ema_20")},
            {"kind": "ema", "length": 50, "col_names": ("ema_50")},
            {"kind": "ema", "length": 200, "col_names": ("ema_200")},
            {"kind": "macd", "fast": 8, "slow": 21, "col_names": ("macd", "macdh", "macds")},
            {"kind": "tema", "length": 5, "col_names": ("tema")},
            {"kind": "kama", "length": 5, "col_names": ("kama")},
            {"kind": "wma", "length": 5, "col_names": ("wma")},
        ]
    )

    data.ta.strategy(CustomStrategy)
    data.dropna(inplace=True)


def get_data(df, window_size, model_name):
    if model_name == '2D-CNN_PI':
        data = df.loc[:, ['volume', 'rsi', 'stochk', 'stochd', 'cci', 'willr', 'mom', 'roc', 'trix', 'trixs',
                          'cmo']].values
    elif model_name == '2D-CNN_CI':
        data = df.loc[:, ['sma_50', 'sma_200', 'ema_10', 'ema_20', 'ema_50', 'ema_200', 'macd', 'macdh', 'macds',
                          'tema', 'kama', 'wma']].values
    else:
        data = df.loc[:, ['volume', 'rsi', 'stochk', 'stochd', 'cci', 'willr', 'mom', 'roc', 'trix', 'trixs', 'cmo',
                          'sma_50', 'sma_200', 'ema_10', 'ema_20', 'ema_50', 'ema_200', 'macd', 'macdh', 'macds',
                          'tema', 'kama', 'wma']].values

    min_max_scaler = MinMaxScaler()
    data = min_max_scaler.fit_transform(data)

    x = []
    for i in range(window_size, data.shape[0]):
        x.append(data[i - window_size:i])

    x = np.expand_dims(x, axis=-1)

    labels = get_labels(df, window_size)
    y = np.array(labels)
    return x, y


def correlation(df):
    corr = df.drop(columns=["open_norm", "high_norm", "low_norm", "close_norm", "action"]).corr()
    fig = plt.figure(figsize=(20, 30))
    ax = fig.add_subplot(111)
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True

    sns.heatmap(corr,
                xticklabels=corr.columns,
                yticklabels=corr.columns,
                ax=ax, mask=mask, square=True,
                cbar_kws={"shrink": 0.2},
                cmap="coolwarm")
    plt.show()


def make_investment(data, action_list, action_name, window_size):
    code_to_action = {0: 'buy', 1: 'None', 2: 'sell'}
    data[action_name] = 'None'
    i = window_size
    for a in action_list:
        data[action_name][i] = code_to_action[a]
        i += 1


# save actions to file
def save_action(action_list, path, file_name):
    directory = f'Results/actions/{path}'

    if not os.path.exists(directory):
        os.makedirs(directory)

    file_path = os.path.join(directory, file_name)
    array_str = ",".join(map(str, action_list))

    with open(file_path, "w") as file:
        file.write(array_str)


# load actions from file
def load_action(file_name):
    directory = f'Results/actions/'

    file_path = os.path.join(directory, file_name)
    try:
        with open(file_path, 'r') as file:
            data_str = file.read().strip().split(',')
    except FileNotFoundError:
        print("Please run DQN, 2D-CNN-PI and 2D-CNN-CI models first!")
        sys.exit(1)

    data_array = []

    for value in data_str:
        try:
            data_array.append(int(value))
        except ValueError:
            pass

    data_array = np.array(data_array)
    return data_array


def get_labels(data, window_size):
    label_threshold = 0.005
    data = data[window_size:]
    ensemble_y_true = np.ones(len(data), dtype=int)
    last_action = 1

    for i in range(len(data) - 1):
        current_price = data.iloc[i]['close']
        next_price = data.iloc[i + 1]['close']

        changes = (next_price - current_price) / current_price

        if changes >= label_threshold or (last_action == Action.BUY.value and 0 <= changes < label_threshold):
            last_action = Action.BUY.value
            ensemble_y_true[i] = Action.BUY.value
        elif changes < -label_threshold or (last_action == Action.SELL.value and 0 > changes >= -label_threshold):
            last_action = Action.SELL.value
            ensemble_y_true[i] = Action.SELL.value
        else:
            last_action = Action.NONE.value
            ensemble_y_true[i] = Action.NONE.value
    return np.array(ensemble_y_true)


def hard_voting(actions):
    final_actions = []

    for row in actions:
        buy_signal_count = 0
        sell_signal_count = 0
        hold_signal_count = 0

        for action in row:
            if action == Action.BUY.value:
                buy_signal_count += 1
            elif action == Action.SELL.value:
                sell_signal_count += 1
            else:
                hold_signal_count += 1

        final_action = Action.NONE.value
        if buy_signal_count > sell_signal_count and buy_signal_count > hold_signal_count:
            final_action = Action.BUY.value
        elif sell_signal_count > buy_signal_count and sell_signal_count > hold_signal_count:
            final_action = Action.SELL.value
        final_actions.append(final_action)

    return final_actions


def evaluate(data, action_pred, dataset_name, model_name, window_size):
    make_investment(data, action_pred, 'action_model', window_size)

    directory = f'Results/{dataset_name}'
    if not os.path.exists(directory):
        os.makedirs(directory)

    path = model_name

    file_name = os.path.join(directory, path)
    ev_model = Evaluation(data, 'action_model', file_name, 1000, 0)
    ev_model.evaluate()
    return ev_model
