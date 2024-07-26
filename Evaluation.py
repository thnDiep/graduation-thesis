import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Evaluation:
    def __init__(self, data, action_label, filename, initial_balance=1000, trading_cost_ratio=0.001):
        self.data = data.loc[:, ['close', action_label, 'action']]
        self.action_label = action_label
        self.filename = filename
        self.trading_cost_ratio = trading_cost_ratio
        self.stop_loss_ratio = 0.015  # 1.5%
        self.take_profit_ratio = 0.02  # 2%

        self.initial_balance = initial_balance
        self.current_balance = initial_balance

        self.total_trade_number = 0
        self.winning_trade_number = 0
        self.losing_trade_number = 0

        self.trades = None
        self.results = []

    def evaluate(self):
        self.init()

        pnl, profit, loss = self.seperate_profit_and_loss()

        self.total_trade_number = len(pnl)
        self.winning_trade_number = len(profit)
        self.losing_trade_number = len(loss)

        if self.losing_trade_number == 0:
            win_loss_ratio = float('inf')
        else:
            win_loss_ratio = self.winning_trade_number / self.losing_trade_number

        total_profit = sum(profit)
        total_loss = abs(sum(loss))

        # Profit factor (PF) = abs(lợi nhuận / lỗ) -> thước đo lợi nhuận, nếu PF < 1 thì lợi nhuận < lỗ
        if total_loss == 0:
            profit_factor = float('inf')
        else:
            profit_factor = total_profit / total_loss

        # Net profit (NP) = lợi nhuận - lỗ -> tổng lợi nhuận thu được từ tất cả các giao dịch
        net_profit = total_profit - total_loss

        # Number of trades (#T) -> số lượng giao dịch đã được thực hiện
        if self.total_trade_number == 0:
            percent_profitable = 0
            average_profit_per_trade = 0
        else:
            # Percent Profitable (PP) = Số giao dịch có lợi nhuận / tổng số giao dịch * 100 -> tỷ lợi phần trăm lợi nhuận
            percent_profitable = (self.winning_trade_number / self.total_trade_number) * 100

            # Average profit per trade (T) =  NP / #T -> lợi nhuận trung bình theo số lượng giao dịch
            average_profit_per_trade = net_profit / self.total_trade_number

        self.results = [self.initial_balance, self.current_balance, net_profit,
                        self.total_trade_number, self.winning_trade_number, self.losing_trade_number,
                        profit_factor, percent_profitable, average_profit_per_trade]

        filename = self.filename + '.txt'

        with open(filename, 'a') as log:
            log.truncate(0)
            log.write(f'Initial balance: {self.initial_balance:.2f}\n')
            log.write(f'Final balance: {self.current_balance:.2f}\n')
            log.write(f'Net Profit (NP): {net_profit:.2f}\n')
            log.write(f'Number of trades (#T): {self.total_trade_number}\n')
            log.write(f'Number of winning trades: {self.winning_trade_number}\n')
            log.write(f'Number of losing trades: {self.losing_trade_number}\n')
            log.write(f'Win/Loss: {win_loss_ratio:.2f}\n')
            log.write(f'Profit factor (PF): {profit_factor:.2f}\n')
            log.write(f'Percent Profitable (PP): {percent_profitable:.2f} %\n')
            log.write(f'Average Profit per trade (T): {average_profit_per_trade:.2f}\n')
            log.write(f'Shape ratio (SR): {self.sharp_ratio():.2f}\n')
            log.write(f'Sortino ratio (SorR): {self.sortino_ratio():.2f}\n')
            log.write(f'Maximum drawdown (MDD): {self.maximum_drawdown():.2f}\n')

    def init(self):
        portfolio = []
        num_shares = 0
        entry_point = None
        entry_close = 0
        entry_balance = 0
        buy_price = 0

        info = []
        portfolio_per_trade = []

        for i in range(len(self.data)):
            action = self.data.iloc[i][self.action_label]
            current_price = self.data.iloc[i]['close']
            if (action == 'sell' or action == 'None') and num_shares == 0:
                self.data.iloc[i, self.data.columns.get_loc('action')] = 'NONE'
                continue

            # Vào lệnh
            if action == 'buy' and num_shares == 0:  # thực hiện giao dịch mua
                entry_point = self.data.index[i]
                entry_close = current_price
                entry_balance = self.current_balance
                num_shares = self.current_balance * (1 - self.trading_cost_ratio) / current_price
                buy_price = num_shares * current_price
                self.current_balance -= buy_price

                # self.total_trades += 1
                if i + 1 < len(self.data):
                    portfolio.append(num_shares * self.data.iloc[i + 1]['close'])

            # Thoát lệnh dựa vào chính sách của mô hình
            elif action == 'sell' and num_shares > 0:  # thực hiện giao dịch bán
                sell_price = num_shares * current_price * (1 - self.trading_cost_ratio)
                profit = sell_price - buy_price

                self.current_balance += sell_price
                # Thời điểm vào lệnh - số dư trước khi vào lệnh - giá đóng cửa tại thời điểm vào lệnh - giá mua
                # Thời điểm thoát lệnh - số dư sau khi thoát lệnh - giá đóng cửa tại thời điểm thoát lệnh - giá bán
                # khối lượng - lợi nhuận
                info.append([entry_point, entry_balance, entry_close, buy_price,
                             self.data.index[i], self.current_balance, current_price, sell_price,
                             num_shares, profit])

                # portfolio cho mỗi giao dịch
                portfolio_per_trade.append(portfolio)

                portfolio = []
                num_shares = 0

            # kiểm tra ngưỡng take-profit và stop-loss
            elif (action == 'None' or action == 'buy') and num_shares > 0:
                isSold = False
                sell_price = num_shares * current_price * (1 - self.trading_cost_ratio)

                profit = sell_price - buy_price
                profit_ratio = profit / buy_price

                if profit_ratio >= self.take_profit_ratio:
                    self.data.iloc[i, self.data.columns.get_loc('action')] = 'Take profit'
                    isSold = True
                elif profit_ratio <= -self.stop_loss_ratio:
                    self.data.iloc[i, self.data.columns.get_loc('action')] = 'Stop loss'
                    isSold = True

                # Thoát lệnh dựa vào chiến lược take profit & stop loss
                if isSold:
                    self.current_balance += sell_price
                    info.append([entry_point, entry_balance, entry_close, buy_price,
                                 self.data.index[i], self.current_balance, current_price, sell_price,
                                 num_shares, profit])
                    portfolio_per_trade.append(portfolio)

                    portfolio = []
                    num_shares = 0
                else:
                    portfolio.append(portfolio[-1] + profit)

            # nếu đang giữ cổ phiếu trong khi tập dữ liệu hết thì hoàn lại giao dịch mua đã thực hiện
            if (i == len(self.data) - 1) and num_shares > 0:
                num_shares = 0
                portfolio = []

                self.current_balance += buy_price

        # Thời điểm vào lệnh - số dư trước khi vào lệnh - giá đóng cửa tại thời điểm vào lệnh - giá mua
        # Thời điểm thoát lệnh - số dư sau khi thoát lệnh - giá đóng cửa tại thời điểm thoát lệnh - giá bán
        # khối lượng - lợi nhuận
        self.trades = pd.DataFrame(info, columns=['entry point', 'entry balance', 'entry close', 'buy price',
                                                  'exit point', 'exit balance', 'exit close', 'sell price',
                                                  'volume', 'profit'])

        # filename = self.filename + '.csv'
        # self.trades.to_csv(filename, index=True)

    def seperate_profit_and_loss(self):
        pnl = self.trades.loc[:, 'profit'].values
        profit = [profit for profit in pnl if profit > 0]
        loss = [profit for profit in pnl if profit < 0]

        return pnl, profit, loss

    def plot_trading_process(self):
        plt.scatter(self.trades['entry point'], self.trades['entry close'], color='green', marker='^', label='Buy')
        plt.scatter(self.trades['exit point'], self.trades['exit close'], color='red', marker='v', label='Sell')
        plt.plot(self.data.index, self.data.close, label='Close price', linestyle='--')

        plt.ylabel('Close Price')
        plt.legend()
        plt.show()

    def get_rate_of_return(self):
        portfolio = self.get_daily_portfolio_value()
        rate_of_return = [(portfolio[p + 1] - portfolio[p]) / portfolio[p] for p in range(len(portfolio) - 1)]
        return rate_of_return

    def get_daily_portfolio_value(self):
        portfolio_value = [self.initial_balance]
        self.arithmetic_return()

        arithmetic_return = self.data[f'arithmetic_daily_return_{self.action_label}'] / 100
        num_shares = 0

        for i in range(len(self.data)):
            action = self.data[self.action_label][i]
            if action == 'buy' and num_shares == 0:  # then buy and pay the transaction cost
                num_shares = portfolio_value[-1] * (1 - self.trading_cost_ratio) / \
                             self.data.iloc[i]['close']
                if i + 1 < len(self.data):
                    portfolio_value.append(num_shares * self.data.iloc[i + 1]['close'])

            elif action == 'sell' and num_shares > 0:  # then sell and pay the transaction cost
                portfolio_value.append(num_shares * self.data.iloc[i]['close'] * (1 - self.trading_cost_ratio))
                num_shares = 0

            elif (action == 'None' or action == 'buy') and num_shares > 0:  # hold shares and get profit
                profit = arithmetic_return[i] * portfolio_value[len(portfolio_value) - 1]
                portfolio_value.append(portfolio_value[-1] + profit)

            elif (action == 'sell' or action == 'None') and num_shares == 0:
                portfolio_value.append(portfolio_value[-1])

        return portfolio_value

    def arithmetic_return(self):
        self.data[f'arithmetic_daily_return_{self.action_label}'] = 0.0

        own_share = False
        for i in range(len(self.data)):
            if (self.data[self.action_label][i] == 'buy') or (own_share and self.data[self.action_label][i] == 'None'):
                own_share = True
                if i < len(self.data) - 1:
                    self.data[f'arithmetic_daily_return_{self.action_label}'][i] = (self.data['close'][i + 1] -
                                                                                    self.data['close'][i]) / \
                                                                                   self.data['close'][i]

            elif self.data[self.action_label][i] == 'sell':
                own_share = False

        self.data[f'arithmetic_daily_return_{self.action_label}'] = self.data[
                                                                        f'arithmetic_daily_return_{self.action_label}'] * 100

    def sharp_ratio(self):
        """
        TODO: We may need to add Risk Free Value in the future
        TODO: How to calculate Risk Free amount?
        https://www.investopedia.com/terms/s/sharperatio.asp

        Since we always have risk, we emit Risk Free Value from the formula

        Subtracting the risk-free rate from the mean return allows an investor to better isolate the profits associated
        with risk-taking activities. Generally, the greater the value of the Sharpe ratio,
        the more attractive the risk-adjusted return.

        The Sharpe ratio can also help explain whether a portfolio's excess returns are due to smart investment decisions
        or a result of too much risk. Although one portfolio or fund can enjoy higher returns than its peers, it is only
        a good investment if those higher returns do not come with an excess of additional risk.

        The greater a portfolio's Sharpe ratio, the better its risk-adjusted-performance. If the analysis results in a
        negative Sharpe ratio, it either means the risk-free rate is greater than the portfolio’s return, or the
        portfolio's return is expected to be negative. In either case, a negative Sharpe ratio does not convey any
        useful meaning.

        :return:
        """
        rate_of_return = self.get_rate_of_return()
        # print(rate_of_return)
        return np.mean(rate_of_return) / np.std(rate_of_return)

    def sortino_ratio(self):
        rate_of_return = self.get_rate_of_return()
        downside_returns = [ret for ret in rate_of_return if ret < 0]
        downside_deviation = np.std(downside_returns) if len(downside_returns) > 0 else 0
        sortino_ratio = np.mean(rate_of_return) / downside_deviation if downside_deviation != 0 else 0

        return sortino_ratio

    def maximum_drawdown(self):
        rate_of_return = self.get_rate_of_return()
        peak_value = np.maximum.accumulate(rate_of_return)
        trough_value = np.minimum.accumulate(rate_of_return)
        mdd = (peak_value[-1] - trough_value[-1]) / peak_value[-1]
        return mdd
