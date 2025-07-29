import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

def directional_accuracy(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    true_direction = np.sign(y_true)
    pred_direction = np.sign(y_pred)
    correct = np.sum(true_direction == pred_direction)
    total = len(y_true)
    return correct / total

def evaluate_model(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    da = directional_accuracy(y_true, y_pred)

    print(f"\nMean Squared Error: {mse:.4f}")
    print(f"R2 Score: {r2:.4f}")
    print(f"Directional Accuracy: {da:.4f}")

    # Plot 1: Actual vs Predicted
    plt.figure(figsize=(10, 4))
    plt.plot(y_true, label='Actual')
    plt.plot(y_pred, label='Predicted')
    plt.title('Actual vs Predicted Returns')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot 2: Residuals
    residuals = np.array(y_true) - np.array(y_pred)
    plt.figure(figsize=(8, 3))
    plt.hist(residuals, bins=50, alpha=0.7)
    plt.title('Prediction Residuals')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot 3: Cumulative Returns
    plt.figure(figsize=(10, 4))
    plt.plot(np.cumsum(y_true), label='Cumulative Actual Returns')
    plt.plot(np.cumsum(y_pred), label='Cumulative Predicted Returns')
    plt.title('Cumulative Returns Comparison')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
