from .Data import Data
import numpy as np
from utils import StateMode


class DataSRExtraction(Data):
    def __init__(self, data, state_mode, dataset_name, data_type, action_name, device, gamma,
                 n_step=4, batch_size=50, window_size=1, transaction_cost=0.0):
        start_index_reward = 0 if state_mode == StateMode.OHLC else window_size - 1
        super().__init__(data, action_name, device, gamma, n_step, batch_size,
                         start_index_reward=start_index_reward,
                         transaction_cost=transaction_cost)

        self.state_mode = state_mode
        self.dataset_name = dataset_name
        self.data_type = data_type
        self.data_kind = 'sr'

        # Windowed + SR percent (column)
        if state_mode == StateMode.SR_PERCENT_COL:
            self.state_size = window_size * 6
            ohlc_array = data.loc[:, ['open_norm', 'high_norm', 'low_norm', 'close_norm',
                                      'support_percent', 'resistance_percent']].values

            for i in range(window_size, ohlc_array.shape[0]):
                temp_states = ohlc_array[i - window_size: i]
                temp_states = temp_states.reshape(-1)
                self.states.append(temp_states)

        # Windowed + SR signal (column)
        elif state_mode == StateMode.SR_SIGNAL_COL:
            self.state_size = window_size * 6
            ohlc_array = data.loc[:, ['open_norm', 'high_norm', 'low_norm', 'close_norm',
                                      'near_support', 'near_resistance']].values

            for i in range(window_size, ohlc_array.shape[0]):
                temp_states = ohlc_array[i - window_size: i]
                temp_states = temp_states.reshape(-1)
                self.states.append(temp_states)

        # Windowed + SR percent (row)
        elif state_mode == StateMode.SR_PERCENT_ROW:
            self.state_size = (window_size + 2) * 4
            ohlc_array = data.loc[:, ['open_norm', 'high_norm', 'low_norm', 'close_norm']].values
            data_support = data.loc[9:, ['support_percent']].values
            data_resistance = data.loc[9:, ['resistance_percent']].values

            for i in range(window_size, ohlc_array.shape[0]):
                ohlc = ohlc_array[i - window_size: i]
                support = np.full(4, data_support[i - window_size])
                resistance = np.full(4, data_resistance[i - window_size])
                temp_states = np.vstack([ohlc, support, resistance])
                temp_states = temp_states.reshape(-1)
                self.states.append(temp_states)

        # Windowed + SR signal (row)
        elif state_mode == StateMode.SR_SIGNAL_ROW:
            self.state_size = (window_size + 2) * 4
            ohlc_array = data.loc[:, ['open_norm', 'high_norm', 'low_norm', 'close_norm']].values
            data_support = data.loc[9:, ['near_support']].values
            data_resistance = data.loc[9:, ['near_resistance']].values

            for i in range(window_size, ohlc_array.shape[0]):
                ohlc = ohlc_array[i - window_size: i]
                support = np.full(4, data_support[i - window_size])
                resistance = np.full(4, data_resistance[i - window_size])
                temp_states = np.vstack([ohlc, support, resistance])
                temp_states = temp_states.reshape(-1)
                self.states.append(temp_states)

        # Windowed + multi time frame SR percent (row)
        elif state_mode == StateMode.MULTI_TIME_FRAME_SR_PERCENT_ROW:
            self.data_kind = 'multi_time_sr'
            self.state_size = (window_size + 6) * 4
            ohlc_array = data.loc[:, ['open_norm', 'high_norm', 'low_norm', 'close_norm']].values
            data_support = data.loc[9:, ['support_percent']].values
            data_resistance = data.loc[9:, ['resistance_percent']].values
            data_support_4h = data.loc[9:, ['support_percent_30m']].values
            data_resistance_4h = data.loc[9:, ['resistance_percent_30m']].values
            data_support_1d = data.loc[9:, ['support_percent_2h']].values
            data_resistance_1d = data.loc[9:, ['resistance_percent_2h']].values

            for i in range(window_size, ohlc_array.shape[0]):
                ohlc = ohlc_array[i - window_size: i]
                support = np.full(4, data_support[i - window_size])
                resistance = np.full(4, data_resistance[i - window_size])
                support_4h = np.full(4, data_support_4h[i - window_size])
                resistance_4h = np.full(4, data_resistance_4h[i - window_size])
                support_1d = np.full(4, data_support_1d[i - window_size])
                resistance_1d = np.full(4, data_resistance_1d[i - window_size])
                temp_states = np.vstack([ohlc, support, resistance, support_4h, resistance_4h, support_1d, resistance_1d])
                temp_states = temp_states.reshape(-1)
                self.states.append(temp_states)

        # Windowed + multi time frame SR signal (row)
        elif state_mode == StateMode.MULTI_TIME_FRAME_SR_SIGNAL_ROW:
            self.data_kind = 'multi_time_sr'
            self.state_size = (window_size + 6) * 4
            ohlc_array = data.loc[:, ['open_norm', 'high_norm', 'low_norm', 'close_norm']].values
            data_support = data.loc[9:, ['near_support']].values
            data_resistance = data.loc[9:, ['near_resistance']].values
            data_support_4h = data.loc[9:, ['near_support_30m']].values
            data_resistance_4h = data.loc[9:, ['near_resistance_30m']].values
            data_support_1d = data.loc[9:, ['near_support_2h']].values
            data_resistance_1d = data.loc[9:, ['near_resistance_2h']].values

            for i in range(window_size, ohlc_array.shape[0]):
                ohlc = ohlc_array[i - window_size: i]
                support = np.full(4, data_support[i - window_size])
                resistance = np.full(4, data_resistance[i - window_size])
                support_4h = np.full(4, data_support_4h[i - window_size])
                resistance_4h = np.full(4, data_resistance_4h[i - window_size])
                support_1d = np.full(4, data_support_1d[i - window_size])
                resistance_1d = np.full(4, data_resistance_1d[i - window_size])
                temp_states = np.vstack([ohlc, support, resistance, support_4h, resistance_4h, support_1d, resistance_1d])
                temp_states = temp_states.reshape(-1)
                self.states.append(temp_states)
