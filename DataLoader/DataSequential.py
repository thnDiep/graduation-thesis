from .Data import Data
import torch
from utils import StateMode
import numpy as np


class DataSequential(Data):
    def __init__(self, data, state_mode, dataset_name, data_type, action_name, device, gamma,
                 n_step=4, batch_size=50, window_size=20, transaction_cost=0.0):

        super().__init__(data, action_name, device, gamma, n_step, batch_size,
                         start_index_reward=(window_size - 1),
                         transaction_cost=transaction_cost)

        self.state_mode = state_mode
        self.dataset_name = dataset_name
        self.data_type = data_type
        self.data_kind = 'sequential_sr'

        if state_mode == StateMode.WINDOWED:  # OHLC
            self.data_kind = 'sequential_ohlc'
            self.state_size = 4
            self.data_preprocessed = data.loc[:, ['open_norm', 'high_norm', 'low_norm', 'close_norm']].values

            # We ignore the first window_size elements of the data because of trend
            # for i in range(window_size - 1, len(self.data_preprocessed) - window_size + 1):
            for i in range(0, len(self.data_preprocessed) - window_size + 1):
                temp_states = torch.zeros(window_size, self.state_size, device=device)
                for j in range(i, i + window_size):
                    temp_states[j - i] = torch.tensor(
                        self.data_preprocessed[j], dtype=torch.float, device=device)

                self.states.append(temp_states.unsqueeze(1))

        elif state_mode == StateMode.SR_PERCENT_COL:
            self.state_size = 6
            self.data_preprocessed = data.loc[:, ['open_norm', 'high_norm', 'low_norm', 'close_norm',
                                                  'support_percent',
                                                  'resistance_percent']].values

            for i in range(0, len(self.data_preprocessed) - window_size):
                temp_states = torch.zeros(window_size, self.state_size, device=device)
                for j in range(i, i + window_size):
                    temp_states[j - i] = torch.tensor(
                        self.data_preprocessed[j], dtype=torch.float, device=device)

                self.states.append(temp_states.unsqueeze(1))

        elif state_mode == StateMode.SR_SIGNAL_COL:
            self.state_size = 6
            self.data_preprocessed = data.loc[:, ['open_norm', 'high_norm', 'low_norm', 'close_norm',
                                                  'near_support',
                                                  'near_resistance']].values

            for i in range(0, len(self.data_preprocessed) - window_size):
                temp_states = torch.zeros(window_size, self.state_size, device=device)
                for j in range(i, i + window_size):
                    temp_states[j - i] = torch.tensor(
                        self.data_preprocessed[j], dtype=torch.float, device=device)

                self.states.append(temp_states.unsqueeze(1))

        elif state_mode == StateMode.SR_PERCENT_ROW:
            self.state_size = 4
            self.data_preprocessed = data.loc[:, ['open_norm', 'high_norm', 'low_norm', 'close_norm']].values
            data_support = data.loc[9:, ['support_percent']].values
            data_resistance = data.loc[9:, ['resistance_percent']].values

            for i in range(0, len(self.data_preprocessed) - window_size + 1):
                temp_states = torch.zeros(window_size + 2, self.state_size, device=device)
                for j in range(i, i + window_size):
                    temp_states[j - i] = torch.tensor(
                        self.data_preprocessed[j], dtype=torch.float, device=device)

                support = np.full(4, data_support[i])
                temp_states[window_size] = torch.tensor(support, dtype=torch.float, device=device)

                resistance = np.full(4, data_resistance[i])
                temp_states[window_size + 1] = torch.tensor(resistance, dtype=torch.float, device=device)

                self.states.append(temp_states.unsqueeze(1))

        elif state_mode == StateMode.SR_SIGNAL_ROW:
            self.state_size = 4
            self.data_preprocessed = data.loc[:, ['open_norm', 'high_norm', 'low_norm', 'close_norm']].values
            data_support = data.loc[9:, ['near_support']].values
            data_resistance = data.loc[9:, ['near_resistance']].values

            for i in range(0, len(self.data_preprocessed) - window_size + 1):
                temp_states = torch.zeros(window_size + 2, self.state_size, device=device)
                for j in range(i, i + window_size):
                    temp_states[j - i] = torch.tensor(
                        self.data_preprocessed[j], dtype=torch.float, device=device)

                support = np.full(4, data_support[i])
                temp_states[window_size] = torch.tensor(support, dtype=torch.float, device=device)

                resistance = np.full(4, data_resistance[i])
                temp_states[window_size + 1] = torch.tensor(resistance, dtype=torch.float, device=device)

                self.states.append(temp_states.unsqueeze(1))

    def __next__(self):
        if self.index_batch < self.num_batch:
            batch = [s for s in
                     self.states[self.index_batch * self.batch_size: (self.index_batch + 1) * self.batch_size]]
            self.index_batch += 1
            return torch.cat(batch, dim=1)

        raise StopIteration