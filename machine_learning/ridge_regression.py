import numpy as np
from ucimlrepo import fetch_ucirepo

class RidgeRegression:
    def __init__(self, X, y, lambda_=0.1):
        self.X = X  #feature input
        self.y = y  #target values
        self.n = y.shape[0]  #number of samples
        self.d = X.shape[1]  #number of features
        self.lambda_ = lambda_  #regularization strength
    def compute_loss(self, w, b):
        y_pred = self.X @ w + b
        mse = np.mean(np.square(y_pred - self.y))
        regularization = self.lambda_ * np.sum(np.square(w))
        loss = mse + regularization
        return loss
    def compute_gradient(self, w, b):
        y_pred = self.X @ w + b
        error = y_pred - self.y
        d_w = (2.0 / self.n) * self.X.T @ error
        d_w = d_w + 2.0 * self.lambda_ * w
        d_b = (2.0 / self.n) * np.sum(error)
        return d_w, d_b

class GradientDescent:
    def __init__(self, learning_rate=0.03, epochs=5000):
        self.lr = learning_rate
        self.epochs = epochs
    def optimize(self, loss_func):
        d = loss_func.d
        w = np.random.randn(d) / np.sqrt(d)
        b = np.random.randn() / np.sqrt(d)
        for epoch in range(self.epochs):
            d_w, d_b = loss_func.compute_gradient(w, b)
            w = w - self.lr * d_w
            b = b - self.lr * d_b
            if epoch % 500 == 0:
                loss = loss_func.compute_loss(w, b)
                print(f"Epoch {epoch:5d} | Loss = {loss:.4f}")
        return w, b

if __name__ == "__main__":
    real_estate_valuation = fetch_ucirepo(id=477)
    X_raw = real_estate_valuation.data.features.values
    y_raw = real_estate_valuation.data.targets.values.ravel()
    print("Dataset name:", real_estate_valuation.metadata.name)
    print("Number of samples:", X_raw.shape[0])
    print("Number of original features:", X_raw.shape[1])
    X_raw = X_raw[:, [1, 2]]
    print("Number of selected features:", X_raw.shape[1])
    X = (X_raw - np.mean(X_raw, axis=0)) / np.std(X_raw, axis=0)
    loss = RidgeRegression(
        X,
        y_raw,
        lambda_=0.1
    )
    optimizer = GradientDescent(
        learning_rate=0.03,
        epochs=5000
    )
    w_opt, b_opt = optimizer.optimize(loss)
    print("=== Training Finished ===")
    print("Optimal weight w:", w_opt)
    print("Optimal bias b:", b_opt)
    final_loss = loss.compute_loss(w_opt, b_opt)
    print("Final Ridge Loss:", final_loss)
