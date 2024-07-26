from .Data import Data
import numpy as np
from utils import StateMode, load_action
from sklearn.preprocessing import MinMaxScaler


class DataIndicatorExtraction(Data):
    def __init__(self, data, state_mode, dataset_name, data_type, action_name, device, gamma,
                 n_step=4, batch_size=50, window_size=1, transaction_cost=0.0):
        start_index_reward = 0 if state_mode != StateMode.WINDOWED else window_size - 1
        super().__init__(data, action_name, device, gamma, n_step, batch_size,
                         start_index_reward=start_index_reward,
                         transaction_cost=transaction_cost)

        self.state_mode = state_mode
        self.dataset_name = dataset_name
        self.data_type = data_type
        self.data_kind = 'indicator'

        # Windowed
        if state_mode == StateMode.WINDOWED:
            self.data_kind = 'stock_data'
            self.state_size = window_size * 4
            ohlc_array = data.loc[:, ['open_norm', 'high_norm', 'low_norm', 'close_norm']].values

            for i in range(window_size, ohlc_array.shape[0]):
                temp_states = ohlc_array[i - window_size: i]
                temp_states = temp_states.reshape(-1)
                self.states.append(temp_states)

        # OHLC + Leading indicators
        elif state_mode == StateMode.PI:
            self.state_size = window_size * (11 + 4)

            ohlc = data.loc[:, ['open_norm', 'high_norm', 'low_norm', 'close_norm']].values

            leading = data.loc[:, ['volume', 'rsi', 'stochk', 'stochd', 'cci', 'willr',
                                   'mom', 'roc', 'trix', 'trixs', 'cmo']].values
            min_max_scaler = MinMaxScaler()
            leading = min_max_scaler.fit_transform(leading)

            ohlc_array = np.concatenate((ohlc, leading), axis=1)

            for i in range(window_size, ohlc_array.shape[0]):
                temp_states = ohlc_array[i - window_size: i]
                temp_states = temp_states.reshape(-1)
                self.states.append(temp_states)

        # OHLC + Lagging indicators
        elif state_mode == StateMode.CI:
            self.state_size = window_size * (12 + 4)

            ohlc = data.loc[:, ['open_norm', 'high_norm', 'low_norm', 'close_norm']].values

            lagging = data.loc[:, ['sma_50', 'sma_200',
                                   'ema_10', 'ema_20', 'ema_50', 'ema_200',
                                   'macd', 'macdh', 'macds',
                                   'tema', 'kama', 'wma']].values
            min_max_scaler = MinMaxScaler()
            lagging = min_max_scaler.fit_transform(lagging)

            ohlc_array = np.concatenate((ohlc, lagging), axis=1)

            for i in range(window_size, ohlc_array.shape[0]):
                temp_states = ohlc_array[i - window_size: i]
                temp_states = temp_states.reshape(-1)
                self.states.append(temp_states)

        # OHLC + Leading + Lagging indicators
        elif state_mode == StateMode.TI:
            self.state_size = window_size * (23 + 4)

            ohlc = data.loc[:, ['open_norm', 'high_norm', 'low_norm', 'close_norm']].values

            indicator = data.loc[:, ['volume', 'rsi', 'stochk', 'stochd', 'cci', 'willr',
                                     'mom', 'roc', 'trix', 'trixs', 'cmo',
                                     'sma_50', 'sma_200',
                                     'ema_10', 'ema_20', 'ema_50', 'ema_200',
                                     'macd', 'macdh', 'macds',
                                     'tema', 'kama', 'wma']].values

            min_max_scaler = MinMaxScaler()
            indicator = min_max_scaler.fit_transform(indicator)

            ohlc_array = np.concatenate((ohlc, indicator), axis=1)

            for i in range(window_size, ohlc_array.shape[0]):
                temp_states = ohlc_array[i - window_size: i]
                temp_states = temp_states.reshape(-1)
                self.states.append(temp_states)

        # OHLC + Leading + Lagging indicator signals from 2D-CNN
        elif state_mode == StateMode.TI_SIGNAL:
            self.state_size = 6 * window_size
            action_leading = load_action(f'{self.dataset_name}/{self.data_type}/2D-CNN_PI')
            action_lagging = load_action(f'{self.dataset_name}/{self.data_type}/2D-CNN_CI')

            ohlc_array = data.loc[:, ['open_norm', 'high_norm', 'low_norm', 'close_norm']].values
            actions = np.vstack((np.ones(ohlc_array.shape[0]), np.ones(ohlc_array.shape[0]))).T
            ohlc_array = np.hstack((ohlc_array, actions))

            for i in range(window_size, ohlc_array.shape[0]):
                temp_states = ohlc_array[i - window_size: i]
                temp_states[-1][4] = action_leading[i - window_size]
                temp_states[-1][5] = action_lagging[i - window_size]
                temp_states = temp_states.reshape(-1)
                self.states.append(temp_states)
