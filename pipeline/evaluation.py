import numpy as np

class Evaluator:
    def sharpe_ratio(self, returns, risk_free=0.0):
        excess = returns - risk_free
        return np.mean(excess) / np.std(excess) * np.sqrt(52)

    def max_drawdown(self, cum_returns):
        peak = np.maximum.accumulate(cum_returns)
        drawdown = (cum_returns - peak) / peak
        return drawdown.min()

    def annualized_return(self, returns):
        return np.mean(returns) * 52

    def annualized_volatility(self, returns):
        return np.std(returns) * np.sqrt(52)
    
    def directional_accuracy(y_true, y_pred):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        # Direction: +1 for positive return, -1 for negative or zero
        true_direction = np.sign(y_true)
        pred_direction = np.sign(y_pred)

        correct = np.sum(true_direction == pred_direction)
        total = len(y_true)

        return correct / total