import warnings
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import os
from pathlib import Path
from utils import add_technical_indicators


class YahooFinanceDataLoader:
    def __init__(self, dataset_name, split_point, begin_date=None, end_date=None, load_from_file=False,
                 multi_frame=False, yahoo_finance=True):
        warnings.filterwarnings('ignore')
        self.DATA_NAME = dataset_name
        self.DATA_PATH = os.path.join(Path(os.path.abspath(os.path.dirname(__file__))).parent,
                                      f'Data/{self.DATA_NAME}/')

        self.DATA_FILE = dataset_name + '.csv'

        self.split_point = split_point
        self.begin_date = begin_date
        self.end_date = end_date
        self.multi_frame = multi_frame
        self.yahoo_finance = yahoo_finance

        if not load_from_file:
            self.data = self.load_data()
            if not self.yahoo_finance:
                self.data = self.data[60:]
            if not self.multi_frame:
                get_sr_from_file(self.data, self.DATA_PATH, self.DATA_NAME)
            else:
                get_multi_time_frame_sr_from_file(self.data, self.DATA_PATH, self.DATA_NAME)
            self.normalize_data()
            self.data.to_csv(f'{self.DATA_PATH}{self.DATA_NAME}_processed.csv', index=True)
        else:
            self.data = pd.read_csv(f'{self.DATA_PATH}{self.DATA_NAME}_processed.csv')
            self.data.set_index('Date', inplace=True)
            self.normalize_data()

        if begin_date is not None:
            self.data = self.data[self.data.index >= begin_date]

        if end_date is not None:
            self.data = self.data[self.data.index <= end_date]

        if type(split_point) == str:
            self.data_train = self.data[self.data.index < split_point]
            self.data_test = self.data[self.data.index >= split_point]
        elif type(split_point) == int:
            self.data_train = self.data[:split_point]
            self.data_test = self.data[split_point:]
        else:
            raise ValueError('Split point should be either int or date!')

        self.data_train_with_date = self.data_train.copy()
        self.data_test_with_date = self.data_test.copy()

        self.data_train.reset_index(drop=True, inplace=True)
        self.data_test.reset_index(drop=True, inplace=True)

        # # open comment the lines below if there is no file image
        self.plot_dataset()
        # self.plot_dataset_with_sr()

        print(f'===== {self.DATA_NAME} =====')
        print("Length: ", len(self.data))
        print("Start date: ", self.data.index[0])
        print("End date: ", self.data.index[-1])
        print("Length of train data: ", len(self.data_train))
        print("Length of test data: ", len(self.data_test))

    def load_data(self):
        if not self.multi_frame:
            data = pd.read_csv(f'{self.DATA_PATH}{self.DATA_FILE}')
            data.dropna(inplace=True)
            data.set_index('Date', inplace=True)
            if self.yahoo_finance:
                data.rename(
                    columns={'Close': 'close', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Volume': 'volume'},
                    inplace=True)
            # # open comment the line below if there is no file low/high_centers
            # calculate_sr(data, self.DATA_PATH, self.DATA_NAME)
            if self.yahoo_finance:
                add_technical_indicators(data)
                data = data.drop(['Adj Close'], axis=1)
            data['action'] = "None"
            return data
        else:
            data_5m = pd.read_csv(f'{self.DATA_PATH}{self.DATA_NAME}.csv')
            data_5m.dropna(inplace=True)
            data_5m.set_index('Date', inplace=True)
            # # open comment the line below if there is no file low/high_centers
            # calculate_sr(data_5m, self.DATA_PATH, f'{self.DATA_NAME}')
            add_technical_indicators(data_5m)
            data_5m['action'] = "None"

            # data_30m = pd.read_csv(f'{self.DATA_PATH}{self.DATA_NAME}_30m.csv')
            # data_30m.dropna(inplace=True)
            # data_30m.set_index('Date', inplace=True)
            # # open comment the line below if there is no file low/high_centers
            # calculate_sr(data_30m, self.DATA_PATH, f'{self.DATA_NAME}_30m', slice_data=False)
            #
            # data_2h = pd.read_csv(f'{self.DATA_PATH}{self.DATA_NAME}_2h.csv')
            # data_2h.dropna(inplace=True)
            # data_2h.set_index('Date', inplace=True)
            # # open comment the line below if there is no file low/high_centers
            # calculate_sr(data_30m, self.DATA_PATH, f'{self.DATA_NAME}_2h', slice_data=False)
            return data_5m

    def normalize_data(self):
        min_max_scaler = MinMaxScaler()
        self.data['open_norm'] = min_max_scaler.fit_transform(self.data.open.values.reshape(-1, 1))
        self.data['high_norm'] = min_max_scaler.fit_transform(self.data.high.values.reshape(-1, 1))
        self.data['low_norm'] = min_max_scaler.fit_transform(self.data.low.values.reshape(-1, 1))
        self.data['close_norm'] = min_max_scaler.fit_transform(self.data.close.values.reshape(-1, 1))

    def plot_dataset(self):
        plt.figure(figsize=(9, 5))
        df1 = pd.Series(self.data_train_with_date.close, index=self.data.index)
        df2 = pd.Series(self.data_test_with_date.close, index=self.data.index)
        ax = df1.plot(color='b', label='Train', linewidth=1)
        df2.plot(ax=ax, color='r', label='Test', linewidth=1)
        plt.legend(loc='upper left')
        ax.set(xlabel='Time', ylabel='Close Price')
        step = max(len(df1) // 4, 1)
        plt.xticks(range(0, len(df1), step), df1.index[::step])
        plt.savefig(f'{self.DATA_PATH}{self.DATA_NAME}_image.jpg', dpi=300)

    def plot_dataset_with_sr(self):
        plt.figure(figsize=(10, 5))

        index = self.data_train_with_date.index[-200:]
        close = self.data_train_with_date.close[-200:]
        support = self.data_train_with_date['support_2h'][-200:]
        resistance = self.data_train_with_date['resistance_2h'][-200:]

        close = pd.Series(close, index=index)
        support = pd.Series(support, index=index)
        resistance = pd.Series(resistance, index=index)

        ax = close.plot(color='blue', label='Data', linewidth=1)
        support.plot(ax=ax, color='orange', label='Support', linewidth=1)
        resistance.plot(ax=ax, color='yellow', label='Resistance', linewidth=1)
        ax.legend(['Close price', 'Support 2h', 'Resistance 2h'], loc='upper left')
        ax.set(xlabel='Time', ylabel='Close Price')
        plt.savefig(f'{self.DATA_PATH}{self.DATA_NAME}_image.jpg', dpi=300)


# determine stock support and resistance levels by using unsupervised learning - KMeans
# references: https://github.com/judopro/Stock_Support_Resistance_ML/tree/master
def get_optimum_clusters(df, saturation_point=0.05):
    '''

    :param df: dataframe
    :param saturation_point: The amount of difference we are willing to detect
    :return: clusters with optimum K centers

    This method uses elbow method to find the optimum number of K clusters
    We initialize different K-means with 1..10 centers and compare the inertias
    If the difference is no more than saturation_point, we choose that as K and move on
    '''

    wcss = []
    k_models = []

    size = min(11, len(df.index))
    for i in range(1, size):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(df)
        wcss.append(kmeans.inertia_)
        k_models.append(kmeans)

    # Compare differences in inertias until it's no more than saturation_point
    optimum_k = len(wcss) - 1
    for i in range(0, len(wcss) - 1):
        diff = abs(wcss[i + 1] - wcss[i])
        if diff < saturation_point:
            optimum_k = i
            break

    print("Optimum K is " + str(optimum_k + 1))
    optimum_clusters = k_models[optimum_k]

    return optimum_clusters


# calculate the support resistance values and save them into the low/high_centers files
def calculate_sr(data, data_path, dataset_name, slice_data=True):
    print('--- calculating: ', dataset_name)
    support_levels = []
    resistance_levels = []
    window_size = 60
    num_columns_needed = 10
    if slice_data:
        data = data[140:]

    low_centers_df = pd.DataFrame(
        columns=[f'low_{i}' for i in range(10)])

    high_centers_df = pd.DataFrame(
        columns=[f'high_{i}' for i in range(10)])

    # for i in range(window_size + 1, data.shape[0]):
    for i in range(window_size, data.shape[0] + 1):
        windowed_data = data[i - window_size: i]  # 0 - 199
        windowed_data_index = data.index[i - window_size: i]
        lows = pd.DataFrame(data=windowed_data, index=windowed_data_index, columns=["low"])
        highs = pd.DataFrame(data=windowed_data, index=windowed_data_index, columns=["high"])

        low_clusters = get_optimum_clusters(lows)
        low_centers = low_clusters.cluster_centers_
        low_centers = np.sort(low_centers, axis=0)
        support_levels.append(low_centers[0][0])

        low_centers = low_centers.reshape(-1)
        if len(low_centers) < num_columns_needed:
            num_values_to_add = num_columns_needed - len(low_centers)
            low_centers = np.pad(low_centers, (0, num_values_to_add), mode='constant', constant_values=np.nan)
        low_centers_df.loc[len(low_centers_df)] = low_centers
        print(low_centers)
        high_clusters = get_optimum_clusters(highs)
        high_centers = high_clusters.cluster_centers_
        high_centers = np.sort(high_centers, axis=0)
        resistance_levels.append(high_centers[-1][0])

        high_centers = high_centers.reshape(-1)
        if len(high_centers) < num_columns_needed:
            num_values_to_add = num_columns_needed - len(high_centers)
            high_centers = np.pad(high_centers, (0, num_values_to_add), mode='constant', constant_values=np.nan)
        high_centers_df.loc[len(high_centers_df)] = high_centers

    low_centers_df.to_csv(f'{data_path}{dataset_name}_low_centers_{window_size}.csv', index=False)
    high_centers_df.to_csv(f'{data_path}{dataset_name}_high_centers_{window_size}.csv', index=False)
    return support_levels, resistance_levels


# get support resistance levels from the low/high_centers files which are nearest close price
# + compute distance percent and
def get_sr_from_file(data, data_path, dataset_name):
    support_levels = pd.read_csv(f'{data_path}{dataset_name}_low_centers_60.csv')
    resistance_levels = pd.read_csv(f'{data_path}{dataset_name}_high_centers_60.csv')

    close_price = data.close.values
    support_column = []
    resistance_column = []

    support_percents = []
    resistance_percents = []
    near_support = []
    near_resistance = []

    for i in range(len(close_price)):
        supports = [x for x in support_levels.iloc[i].values if x is not None and not np.isnan(x)]
        resistances = [x for x in resistance_levels.iloc[i].values if x is not None and not np.isnan(x)]

        # get support closest and smaller than close price,
        # if don't have any support <= close price -> get lowest support
        temp_support = supports[-1]
        prev_support = supports[0]
        for support in supports:
            if support >= close_price[i]:
                temp_support = prev_support
                break
            prev_support = support

        # get resistance closest and larger than close price,
        # if don't have any resistance >= close price -> get highest resistance
        temp_resistance = resistances[-1]  # highest
        for resistance in resistances:  # 9 - 10 clusters
            if resistance >= close_price[i]:
                temp_resistance = resistance
                break

        # percent
        support_percent = abs(close_price[i] / temp_support - 1)
        resistance_percent = abs(close_price[i] / temp_resistance - 1)

        # signals
        if (support_percent < 0.0005) or (close_price[i] < temp_support):
            near_support.append(1)
        else:
            near_support.append(0)

        if (resistance_percent < 0.0005) or (close_price[i] > temp_resistance):
            near_resistance.append(1)
        else:
            near_resistance.append(0)

        # get values nearest close price
        support_column.append(temp_support)
        resistance_column.append(temp_resistance)

        support_percents.append(support_percent)
        resistance_percents.append(resistance_percent)

    data['action'] = "None"
    data['support'] = support_column
    data['resistance'] = resistance_column

    data['support_percent'] = support_percents
    data['resistance_percent'] = resistance_percents
    data['near_support'] = near_support
    data['near_resistance'] = near_resistance

    data.to_csv(f'{data_path}{dataset_name}_processed.csv')
    return data


# get support resistance levels from the low/high_centers files which are nearest close price
# + compute distance percent and
# + merge multi_time_frame
def get_multi_time_frame_sr_from_file(data, data_path, dataset_name):
    data.index = pd.to_datetime(data.index)
    get_sr_from_file(data, data_path, f'{dataset_name}')

    data_30m = pd.read_csv(f'{data_path}{dataset_name}_30m.csv')
    data_30m['Date'] = pd.to_datetime(data_30m['Date'])
    data_30m.set_index("Date", inplace=True)
    data_30m = data_30m[60:]
    data_30m = get_sr_from_file(data_30m, data_path, f'{dataset_name}_30m')

    data_2h = pd.read_csv(f'{data_path}{dataset_name}_2h.csv')
    data_2h['Date'] = pd.to_datetime(data_2h['Date'])
    data_2h.set_index("Date", inplace=True)
    data_2h = data_2h[60:]
    data_2h = get_sr_from_file(data_2h, data_path, f'{dataset_name}_2h')

    support_30m = []
    resistance_30m = []
    support_percent_30m = []
    resistance_percent_30m = []
    near_support_30m = []
    near_resistance_30m = []

    support_2h = []
    resistance_2h = []
    support_percent_2h = []
    resistance_percent_2h = []
    near_support_2h = []
    near_resistance_2h = []

    # Go through each row in data 5m and add the corresponding values from data_30m and data_2h
    for index, row in data.iterrows():
        time_5m = index.time()

        datetime_5m = pd.Timestamp.combine(index.date(), time_5m)

        matching_row_30m = data_30m[(data_30m.index <= datetime_5m) & (
                data_30m.index > (datetime_5m - pd.Timedelta(minutes=30)))]

        matching_row_2h = data_2h[(data_2h.index <= datetime_5m) & (
                data_2h.index > (datetime_5m - pd.Timedelta(hours=2)))]

        # Add values from matching rows into the 5m data. If there are no matching rows, add None.
        if not matching_row_30m.empty:
            matching_row_30m = matching_row_30m.iloc[-1]
            support_30m.append(matching_row_30m['support'])
            resistance_30m.append(matching_row_30m['resistance'])
            support_percent_30m.append(matching_row_30m['support_percent'])
            resistance_percent_30m.append(matching_row_30m['resistance_percent'])
            near_support_30m.append(matching_row_30m['near_support'])
            near_resistance_30m.append(matching_row_30m['near_resistance'])
        else:
            support_30m.append(None)
            resistance_30m.append(None)
            support_percent_30m.append(None)
            resistance_percent_30m.append(None)
            near_support_30m.append(None)
            near_resistance_30m.append(None)

        if not matching_row_2h.empty:
            matching_row = matching_row_2h.iloc[-1]
            support_2h.append(matching_row['support'])
            resistance_2h.append(matching_row['resistance'])
            support_percent_2h.append(matching_row['support_percent'])
            resistance_percent_2h.append(matching_row['resistance_percent'])
            near_support_2h.append(matching_row['near_support'])
            near_resistance_2h.append(matching_row['near_resistance'])
        else:
            support_2h.append(None)
            resistance_2h.append(None)
            support_percent_2h.append(None)
            resistance_percent_2h.append(None)
            near_support_2h.append(None)
            near_resistance_2h.append(None)

    data['support_30m'] = support_30m
    data['resistance_30m'] = resistance_30m
    data['support_percent_30m'] = support_percent_30m
    data['resistance_percent_30m'] = resistance_percent_30m
    data['near_support_30m'] = near_support_30m
    data['near_resistance_30m'] = near_resistance_30m

    data['support_2h'] = support_2h
    data['resistance_2h'] = resistance_2h
    data['support_percent_2h'] = support_percent_2h
    data['resistance_percent_2h'] = resistance_percent_2h
    data['near_support_2h'] = near_support_2h
    data['near_resistance_2h'] = near_resistance_2h

    # choose one of two lines below. (drop or fill None data)
    # data.dropna(inplace=True)
    data.fillna(0, inplace=True)

    data = data.drop(['action'], axis=1)
    data['action'] = "None"

    data.to_csv(f'{data_path}{dataset_name}_processed.csv')
