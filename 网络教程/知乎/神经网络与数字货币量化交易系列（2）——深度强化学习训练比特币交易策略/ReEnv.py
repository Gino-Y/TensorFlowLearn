class BitcoinTradingEnv:
    def __init__(self, df, commission=0.00075, initial_balance=10000, initial_stocks=1, all_data=False,
                 sample_length=500):
        self.initial_stocks = initial_stocks  # 初始的比特币数量
        self.initial_balance = initial_balance  # 初始的资产
        self.current_time = 0  # 回测的时间位置
        self.commission = commission  # 易手续费
        self.done = False  # 回测是否结束
        self.df = df
        self.norm_df = 100 * (self.df / self.df.shift(1) - 1).fillna(0)  # 标准化方法，简单的收益率标准化
        self.mode = all_data  # 是否为抽样回测模式
        self.sample_length = 500  # 抽样长度

    def reset(self):
        self.balance = self.initial_balance
        self.stocks = self.initial_stocks
        self.last_profit = 0

        if self.mode:
            self.start = 0
            self.end = self.df.shape[0] - 1
        else:
            self.start = np.random.randint(0, self.df.shape[0] - self.sample_length)
            self.end = self.start + self.sample_length

        self.initial_value = self.initial_balance + self.initial_stocks * self.df.iloc[self.start, 4]
        self.stocks_value = self.initial_stocks * self.df.iloc[self.start, 4]
        self.stocks_pct = self.stocks_value / self.initial_value
        self.value = self.initial_value

        self.current_time = self.start
        return np.concatenate(
            [self.norm_df[['o', 'h', 'l', 'c', 'v']].iloc[self.start].values, [self.balance / 10000, self.stocks / 1]])

    def step(self, action):
        # action即策略采取的动作，这里将更新账户和计算reward
        done = False
        if action == 0:  # 持有
            pass
        elif action == 1:  # 买入
            buy_value = self.balance * 0.5
            if buy_value > 1:  # 余钱不足，不操作账户
                self.balance -= buy_value
                self.stocks += (1 - self.commission) * buy_value / self.df.iloc[self.current_time, 4]
        elif action == 2:  # 卖出
            sell_amount = self.stocks * 0.5
            if sell_amount > 0.0001:
                self.stocks -= sell_amount
                self.balance += (1 - self.commission) * sell_amount * self.df.iloc[self.current_time, 4]

        self.current_time += 1
        if self.current_time == self.end:
            done = True
        self.value = self.balance + self.stocks * self.df.iloc[self.current_time, 4]
        self.stocks_value = self.stocks * self.df.iloc[self.current_time, 4]
        self.stocks_pct = self.stocks_value / self.value
        if self.value < 0.1 * self.initial_value:
            done = True

        profit = self.value - (self.initial_balance + self.initial_stocks * self.df.iloc[self.current_time, 4])
        reward = profit - self.last_profit  # 每回合的reward是新增收益
        self.last_profit = profit
        next_state = np.concatenate([self.norm_df[['o', 'h', 'l', 'c', 'v']].iloc[self.current_time].values,
                                     [self.balance / 10000, self.stocks / 1]])
        return (next_state, reward, done, profit)