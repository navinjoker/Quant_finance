import numpy as np

class PortfolioAllocator:
    def equal_weight(self, n_assets):
        return np.ones(n_assets) / n_assets

    def mean_variance(self, predicted_returns, cov_matrix):
        inv_cov = np.linalg.pinv(cov_matrix)
        weights = inv_cov @ predicted_returns
        return weights / np.sum(weights)

    def risk_parity(self, cov_matrix):
        vol = np.sqrt(np.diag(cov_matrix))
        inv_vol = 1 / vol
        return inv_vol / np.sum(inv_vol)