import numpy as np
import tensorflow as tf
from keras.models import load_model
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier

from DataLoader.DataLoader import YahooFinanceDataLoader
from DataLoader.DataIndicatorExtraction import DataIndicatorExtraction
from DataLoader.DataSRExtraction import DataSRExtraction
from DataLoader.DataSequential import DataSequential
from BaselineModels.VanillaInput.Train import Train as DeepRL
from BaselineSequentialModels.CNN2D.Train import Train as CNN2d
import utils
from utils import StateMode
from CNNModel import cnnpred_2d

DATA_LOADERS = {
    'AAL': YahooFinanceDataLoader('AAL',
                                  split_point='2023-01-01',
                                  load_from_file=True),

    'BTC-USD': YahooFinanceDataLoader('BTC-USD',
                                      split_point='2023-01-01',
                                      load_from_file=True),

    'GE': YahooFinanceDataLoader('GE',
                                 split_point='2023-01-01',
                                 load_from_file=True),

    'GOOGL': YahooFinanceDataLoader('GOOGL',
                                    split_point='2023-01-01',
                                    load_from_file=True),

    'SH1A0001': YahooFinanceDataLoader('SH1A0001',
                                       split_point='2016-10-01',
                                       load_from_file=True,
                                       multi_frame=True),

    'SZ399005': YahooFinanceDataLoader('SZ399005',
                                       split_point='2016-10-01',
                                       load_from_file=True,
                                       multi_frame=True),
}


class SensitivityRun:
    def __init__(self,
                 dataset_name,
                 gamma,
                 batch_size,
                 replay_memory_size,
                 feature_size,
                 target_update,
                 n_episodes,
                 n_step,
                 window_size,
                 device,
                 model_name,
                 transaction_cost=0):
        self.data_loader = DATA_LOADERS[dataset_name]
        self.dataset_name = dataset_name
        self.gamma = gamma
        self.batch_size = batch_size
        self.replay_memory_size = replay_memory_size
        self.feature_size = feature_size
        self.target_update = target_update
        self.n_episodes = n_episodes
        self.n_step = n_step
        self.transaction_cost = transaction_cost
        self.window_size = window_size
        self.device = device
        self.model_name = model_name
        self.state_mode = None

        self.dataTrain = None
        self.dataTest = None

        self.x_train = None
        self.y_train = None

        self.x_test = None
        self.y_test = None

        self.x_valid = None
        self.y_valid = None

        self.dqn_agent = None
        self.cnn_agent = None

        self.cnn2d = None
        self.reference_model = None

        self.early_stopping = None
        self.reset()

    def reset(self):
        self.load_data()
        self.load_agents()

    def load_data(self):
        # LOAD DATA - CNN2D
        if self.model_name == '2D-CNN_PI' or self.model_name == '2D-CNN_CI':
            x_train, y_train = utils.get_data(self.data_loader.data_train,
                                              self.window_size,
                                              self.model_name)

            self.x_test, self.y_test = utils.get_data(self.data_loader.data_test,
                                                      self.window_size,
                                                      self.model_name)

            self.x_train, self.x_valid, self.y_train, self.y_valid = train_test_split(x_train,
                                                                                      y_train,
                                                                                      test_size=0.1,
                                                                                      shuffle=False)

        # LOAD DATA - REFERENCE MODELS (Action from DQN + 2D-CNN_PI + 2D-CNN_CI)
        elif self.model_name == 'RF' \
                or self.model_name == 'SVM' \
                or self.model_name == 'HARD_VOTING':
            train_action_ohlc = utils.load_action(f'{self.dataset_name}/train/DQN')
            train_action_leading = utils.load_action(f'{self.dataset_name}/train/2D-CNN_PI')
            train_action_lagging = utils.load_action(f'{self.dataset_name}/train/2D-CNN_CI')

            self.x_train = np.column_stack((train_action_ohlc, train_action_leading, train_action_lagging))
            _, self.y_train = utils.get_data(self.data_loader.data_train,
                                             self.window_size,
                                             self.model_name)

            test_action_ohlc = utils.load_action(f'{self.dataset_name}/test/DQN')
            test_action_leading = utils.load_action(f'{self.dataset_name}/test/2D-CNN_PI')
            test_action_lagging = utils.load_action(f'{self.dataset_name}/test/2D-CNN_CI')

            self.x_test = np.column_stack((test_action_ohlc, test_action_leading, test_action_lagging))
            _, self.y_test = utils.get_data(self.data_loader.data_test,
                                            self.window_size,
                                            self.model_name)

        # LOAD DATA - DQN AGENT with CNN
        elif self.model_name == 'DQN_CNN' \
                or self.model_name == 'DQN_CNN_SRP_COL' \
                or self.model_name == 'DQN_CNN_SRS_COL' \
                or self.model_name == 'DQN_CNN_SRP_ROW' \
                or self.model_name == 'DQN_CNN_SRS_ROW':
            if self.model_name == 'DQN_CNN':
                self.state_mode = StateMode.WINDOWED
            elif self.model_name == 'DQN_CNN_SRP_COL':
                self.state_mode = StateMode.SR_PERCENT_COL
            elif self.model_name == 'DQN_CNN_SRS_COL':
                self.state_mode = StateMode.SR_SIGNAL_COL
            elif self.model_name == 'DQN_CNN_SRP_ROW':
                self.state_mode = StateMode.SR_PERCENT_ROW
            elif self.model_name == 'DQN_CNN_SRS_ROW':
                self.state_mode = StateMode.SR_SIGNAL_ROW

            self.dataTrain = DataSequential(self.data_loader.data_train,
                                            self.state_mode,
                                            self.dataset_name,
                                            'train',
                                            'action_sr_cnn',
                                            self.device,
                                            self.gamma,
                                            self.n_step,
                                            self.batch_size,
                                            self.window_size,
                                            self.transaction_cost)

            self.dataTest = DataSequential(self.data_loader.data_test,
                                           self.state_mode,
                                           self.dataset_name,
                                           'test',
                                           'action_sr_cnn',
                                           self.device,
                                           self.gamma,
                                           self.n_step,
                                           self.batch_size,
                                           self.window_size,
                                           self.transaction_cost)
        # LOAD DATA - DQN AGENT
        elif self.model_name == 'DQN' \
                or self.model_name == 'DQN_PI' \
                or self.model_name == 'DQN_CI' \
                or self.model_name == 'DQN_TI' \
                or self.model_name == 'DQN_TI_SIGNAL':
            if self.model_name == 'DQN':
                self.state_mode = StateMode.WINDOWED
            elif self.model_name == 'DQN_PI':
                self.state_mode = StateMode.PI
            elif self.model_name == 'DQN_CI':
                self.state_mode = StateMode.CI
            elif self.model_name == 'DQN_TI':
                self.state_mode = StateMode.TI
            elif self.model_name == 'DQN_TI_SIGNAL':
                self.state_mode = StateMode.TI_SIGNAL

            self.dataTrain = DataIndicatorExtraction(self.data_loader.data_train,
                                                     self.state_mode,
                                                     self.dataset_name,
                                                     'train',
                                                     'action_technical',
                                                     self.device,
                                                     self.gamma,
                                                     self.n_step,
                                                     self.batch_size,
                                                     self.window_size,
                                                     self.transaction_cost)

            self.dataTest = DataIndicatorExtraction(self.data_loader.data_test,
                                                    self.state_mode,
                                                    self.dataset_name,
                                                    'test',
                                                    'action_technical',
                                                    self.device,
                                                    self.gamma,
                                                    self.n_step,
                                                    self.batch_size,
                                                    self.window_size,
                                                    self.transaction_cost)

        else:
            if self.model_name == 'DQN_SRP_COL':
                self.state_mode = StateMode.SR_PERCENT_COL
            elif self.model_name == 'DQN_SRS_COL':
                self.state_mode = StateMode.SR_SIGNAL_COL
            elif self.model_name == 'DQN_SRP_ROW':
                self.state_mode = StateMode.SR_PERCENT_ROW
            elif self.model_name == 'DQN_SRS_ROW':
                self.state_mode = StateMode.SR_SIGNAL_ROW
            elif self.model_name == 'DQN_MTF_SRP':
                self.state_mode = StateMode.MULTI_TIME_FRAME_SR_PERCENT_ROW
            elif self.model_name == 'DQN_MTF_SRS':
                self.state_mode = StateMode.MULTI_TIME_FRAME_SR_SIGNAL_ROW

            self.dataTrain = DataSRExtraction(self.data_loader.data_train,
                                              self.state_mode,
                                              self.dataset_name,
                                              'train',
                                              'action_sr',
                                              self.device,
                                              self.gamma,
                                              self.n_step,
                                              self.batch_size,
                                              self.window_size,
                                              self.transaction_cost)

            self.dataTest = DataSRExtraction(self.data_loader.data_test,
                                             self.state_mode,
                                             self.dataset_name,
                                             'test',
                                             'action_sr',
                                             self.device,
                                             self.gamma,
                                             self.n_step,
                                             self.batch_size,
                                             self.window_size,
                                             self.transaction_cost)

    def load_agents(self):
        # LOAD MODEL - CNN2D
        if self.model_name == '2D-CNN_PI' \
                or self.model_name == '2D-CNN_CI':

            num_features = 11 if self.model_name == '2D-CNN_PI' else 12
            self.early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', min_delta=0, patience=100, verbose=0,
                mode='auto', baseline=None, restore_best_weights=False
            )

            self.cnn2d = cnnpred_2d(self.window_size, num_features, [8, 8, 8])
            self.cnn2d.compile(optimizer='adam',
                               loss='sparse_categorical_crossentropy',
                               metrics=['accuracy'])

        # LOAD MODEL - REFERENCE MODELS
        elif self.model_name == 'RF':
            self.reference_model = RandomForestClassifier(n_estimators=200, random_state=42)

        elif self.model_name == 'SVM':
            self.reference_model = svm.SVC(kernel='linear', C=1)

        # LOAD MODEL - DQN AGENT with CNN
        elif self.model_name == 'DQN_CNN' \
                or self.model_name == 'DQN_CNN_SRP_COL' \
                or self.model_name == 'DQN_CNN_SRS_COL':
            self.cnn_agent = CNN2d(self.data_loader,
                                   self.dataTrain,
                                   self.dataTest,
                                   self.dataset_name,
                                   self.feature_size,
                                   self.model_name,
                                   self.state_mode,
                                   self.window_size,
                                   self.transaction_cost,
                                   BATCH_SIZE=self.batch_size,
                                   GAMMA=self.gamma,
                                   ReplayMemorySize=self.replay_memory_size,
                                   TARGET_UPDATE=self.target_update,
                                   n_step=self.n_step)

        elif self.model_name == 'DQN_CNN_SRP_ROW' \
                or self.model_name == 'DQN_CNN_SRS_ROW':
            self.cnn_agent = CNN2d(self.data_loader,
                                   self.dataTrain,
                                   self.dataTest,
                                   self.dataset_name,
                                   self.feature_size,
                                   self.model_name,
                                   self.state_mode,
                                   self.window_size + 2,
                                   self.transaction_cost,
                                   BATCH_SIZE=self.batch_size,
                                   GAMMA=self.gamma,
                                   ReplayMemorySize=self.replay_memory_size,
                                   TARGET_UPDATE=self.target_update,
                                   n_step=self.n_step)

        # LOAD MODEL - DQN AGENT
        elif self.model_name == 'DQN' \
                or self.model_name == 'DQN_PI' \
                or self.model_name == 'DQN_CI' \
                or self.model_name == 'DQN_TI' \
                or self.model_name == 'DQN_TI_SIGNAL' \
                or self.model_name == 'DQN_SRP_COL' \
                or self.model_name == 'DQN_SRS_COL':
            self.dqn_agent = DeepRL(self.data_loader,
                                    self.dataTrain,
                                    self.dataTest,
                                    self.dataset_name,
                                    self.model_name,
                                    self.state_mode,
                                    self.window_size,
                                    self.transaction_cost,
                                    BATCH_SIZE=self.batch_size,
                                    GAMMA=self.gamma,
                                    ReplayMemorySize=self.replay_memory_size,
                                    TARGET_UPDATE=self.target_update,
                                    n_step=self.n_step)

        elif self.model_name == 'DQN_SRP_ROW' \
                or self.model_name == 'DQN_SRS_ROW':
            self.dqn_agent = DeepRL(self.data_loader,
                                    self.dataTrain,
                                    self.dataTest,
                                    self.dataset_name,
                                    self.model_name,
                                    self.state_mode,
                                    self.window_size + 2,
                                    self.transaction_cost,
                                    BATCH_SIZE=self.batch_size,
                                    GAMMA=self.gamma,
                                    ReplayMemorySize=self.replay_memory_size,
                                    TARGET_UPDATE=self.target_update,
                                    n_step=self.n_step)

        elif self.model_name == 'DQN_MTF_SRP' \
                or self.model_name == 'DQN_MTF_SRS':
            self.dqn_agent = DeepRL(self.data_loader,
                                    self.dataTrain,
                                    self.dataTest,
                                    self.dataset_name,
                                    self.model_name,
                                    self.state_mode,
                                    self.window_size + 6,
                                    self.transaction_cost,
                                    BATCH_SIZE=self.batch_size,
                                    GAMMA=self.gamma,
                                    ReplayMemorySize=self.replay_memory_size,
                                    TARGET_UPDATE=self.target_update,
                                    n_step=self.n_step)

    def train(self):
        # TRAIN - CNN2D
        if self.model_name == '2D-CNN_PI' or self.model_name == '2D-CNN_CI':
            self.cnn2d.fit(self.x_train, self.y_train,
                           epochs=200, batch_size=32,
                           callbacks=[self.early_stopping],
                           validation_data=(self.x_valid, self.y_valid),
                           shuffle=False)

            self.cnn2d.save(f'Models/{self.dataset_name}/{self.model_name}', save_format='tf')
            actions = np.concatenate((self.y_train, self.y_valid))
            utils.save_action(actions, f'{self.dataset_name}/train/', self.model_name)

        # TRAIN - REFERENCE MODELS
        elif self.model_name == 'RF' \
                or self.model_name == 'SVM':
            self.reference_model.fit(self.x_train, self.y_train)

        # TRAIN - DQN AGENT with CNN
        elif self.model_name == 'DQN_CNN' \
                or self.model_name == 'DQN_CNN_SRP_COL' \
                or self.model_name == 'DQN_CNN_SRS_COL' \
                or self.model_name == 'DQN_CNN_SRP_ROW' \
                or self.model_name == 'DQN_CNN_SRS_ROW':
            self.cnn_agent.train(self.n_episodes)

        # TRAIN - DQN AGENT
        elif self.model_name == 'DQN' \
                or self.model_name == 'DQN_PI' \
                or self.model_name == 'DQN_CI' \
                or self.model_name == 'DQN_TI' \
                or self.model_name == 'DQN_TI_SIGNAL' \
                or self.model_name == 'DQN_SRP_COL' \
                or self.model_name == 'DQN_SRS_COL' \
                or self.model_name == 'DQN_SRP_ROW' \
                or self.model_name == 'DQN_SRS_ROW' \
                or self.model_name == 'DQN_MTF_SRP' \
                or self.model_name == 'DQN_MTF_SRS':
            action_list = self.dqn_agent.train(self.n_episodes)
            utils.save_action(action_list, f'{self.dataset_name}/train/', self.model_name)

    def test(self):
        # TEST - CNN2D
        if self.model_name == '2D-CNN_PI' or self.model_name == '2D-CNN_CI':
            self.cnn2d = load_model(f'Models/{self.dataset_name}/{self.model_name}')
            y_pred = self.cnn2d.predict(self.x_test)
            action_pred = np.argmax(y_pred, axis=1)

            utils.evaluate(self.data_loader.data_test, action_pred, self.dataset_name,
                           self.model_name, self.window_size)

            utils.save_action(action_pred, f'{self.dataset_name}/test/', self.model_name)
            print(classification_report(action_pred, self.y_test))

        # TEST - REFERENCE MODELS
        elif self.model_name == 'RF' \
                or self.model_name == 'SVM':
            action_pred = self.reference_model.predict(self.x_test)

            utils.evaluate(self.data_loader.data_test, action_pred, self.dataset_name,
                           self.model_name, self.window_size)

            print(classification_report(action_pred, self.y_test))

        elif self.model_name == 'HARD_VOTING':
            action_pred = utils.hard_voting(self.x_test)

            utils.evaluate(self.data_loader.data_test, action_pred, self.dataset_name,
                           self.model_name, self.window_size)

            print(classification_report(action_pred, self.y_test))

        # TEST - DQN AGENT
        elif self.model_name == 'DQN' \
                or self.model_name == 'DQN_PI' \
                or self.model_name == 'DQN_CI' \
                or self.model_name == 'DQN_TI' \
                or self.model_name == 'DQN_TI_SIGNAL' \
                or self.model_name == 'DQN_SRP_COL' \
                or self.model_name == 'DQN_SRS_COL' \
                or self.model_name == 'DQN_SRP_ROW' \
                or self.model_name == 'DQN_SRS_ROW' \
                or self.model_name == 'DQN_MTF_SRP' \
                or self.model_name == 'DQN_MTF_SRS':
            ev_agent, action_list = self.dqn_agent.test()
            # ev_agent.plot_trading_process()
            utils.save_action(action_list, f'{self.dataset_name}/test/', self.model_name)

        # TEST - DQN AGENT with CNN
        elif self.model_name == 'DQN_CNN' \
                or self.model_name == 'DQN_CNN_SRP_COL' \
                or self.model_name == 'DQN_CNN_SRS_COL' \
                or self.model_name == 'DQN_CNN_SRP_ROW' \
                or self.model_name == 'DQN_CNN_SRS_ROW':
            ev_agent, action_list = self.cnn_agent.test()
            # ev_agent.plot_trading_process()
            utils.save_action(action_list, f'{self.dataset_name}/test/', self.model_name)
