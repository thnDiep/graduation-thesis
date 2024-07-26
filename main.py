import argparse
import torch
from utils import set_random_seed
from SensitivityRun import SensitivityRun

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', type=str, default='DQN_TI_SIGNAL',
                    help='[2D-CNN_PI, 2D-CNN_CI, '
                         'DQN, DQN_PI, DQN_CI, DQN_TI, '
                         'DQN_SRP_COL, DQN_SRS_COL, '
                         'DQN_SRP_ROW, DQN_SRS_ROW, '
                         'DQN_MTF_SRP, DQN_MTF_SRS, '
                         'DQN_TI_SIGNAL, '  # Ours
                         'RF, SVM, HARD_VOTING, ' 
                         'DQN_CNN, '
                         'DQN_CNN_SRP_COL, DQN_CNN_SRS_COL, '                         
                         'DQN_CNN_SRP_ROW , DQN_CNN_SRS_ROW]')
parser.add_argument('-t', '--trader', type=str, default='test', help='[train, test, train_test]')
parser.add_argument('-d', '--dataset', default="BTC-USD", help='Name of the data inside the Data folder')
parser.add_argument('-n', '--nep', type=int, default=1, help='Number of episodes')
parser.add_argument('-w', '--window_size', type=int, default=10, help='Window size for sequential models')
parser.add_argument('--cuda', action="store_true", help='run on CUDA (default: False)')
args = parser.parse_args()


if __name__ == '__main__':
    set_random_seed(42)
    n_step = 8
    window_size = args.window_size
    dataset_name = args.dataset
    n_episodes = args.nep
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    print("using: ", device)
    feature_size = 64
    target_update = 5

    gamma = 0.9
    batch_size = 16
    replay_memory_size = 32

    trader = args.trader
    model_name = args.model

    run = SensitivityRun(
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
        transaction_cost=0)
    run.reset()

    if trader == 'train' or trader == 'train_test':
        run.train()

    if trader == 'test' or trader == 'train_test':
        run.test()
