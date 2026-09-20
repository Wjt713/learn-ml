import numpy as np
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class RegularizedLogisticRegression:
    def __init__(self, X, y, reg_strength=0.01):
        self.X = X
        self.y = y
        self.samples = X.shape[0]  # number of training samples
        self.features = X.shape[1]  # number of input features
        self.reg_strength = reg_strength  # L2 regularization strength

    def sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def compute_loss(self, w, b):
        z = self.X @ w + b
        data_loss = np.mean(np.logaddexp(0, z) - self.y * z)
        regularization = self.reg_strength * np.sum(w ** 2)
        return data_loss + regularization

    def compute_gradient(self, w, b):
        scores = self.X @ w + b
        probabilities = self.sigmoid(scores)
        error = probabilities - self.y
        gradient_w = (
            (self.X.T @ error) / self.samples
            + 2.0 * self.reg_strength * w
        )
        gradient_b = np.mean(error)
        return gradient_w, gradient_b


class GradientDescent:
    def __init__(self, step_size=0.05, max_iterations=50000):
        self.step_size = step_size  # learning rate
        self.max_iterations = max_iterations  # number of GD iterations

    def optimize(self, model):
        weights = np.zeros(model.features)
        bias = 0.0
        for iteration in range(self.max_iterations):
            gradient_w, gradient_b = model.compute_gradient(weights, bias)
            weights -= self.step_size * gradient_w
            bias -= self.step_size * gradient_b
            if iteration % 1000 == 0:
                current_loss = model.compute_loss(weights, bias)
                print(
                    f"Iteration {iteration:5d} | "
                    f"Loss = {current_loss:.6f}"
                )
        return weights, bias


def predict(X, weights, bias):
    scores = X @ weights + bias
    scores = np.clip(scores, -500, 500)
    probabilities = 1.0 / (1.0 + np.exp(-scores))
    return (probabilities >= 0.5).astype(int)


if __name__ == "__main__":
    bank_data = fetch_ucirepo(id=222)
    X_raw = bank_data.data.features
    y_raw = bank_data.data.targets
    print("Dataset:", bank_data.metadata.name)
    print("Original samples:", X_raw.shape[0])
    print("Original features:", X_raw.shape[1])
    y = (
        y_raw.iloc[:, 0]
        .astype(str)
        .str.lower()
        .eq("yes")
        .astype(float)
        .to_numpy()
    )
    categorical_features = X_raw.select_dtypes(include=["object"]).columns
    numerical_features = X_raw.select_dtypes(exclude=["object"]).columns
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )
    encoder = OneHotEncoder(
        drop="first",
        handle_unknown="ignore",
        sparse_output=False
    )
    X_train_cat = encoder.fit_transform(X_train_raw[categorical_features])
    X_test_cat = encoder.transform(X_test_raw[categorical_features])
    scaler = StandardScaler()
    X_train_num = scaler.fit_transform(X_train_raw[numerical_features])
    X_test_num = scaler.transform(X_test_raw[numerical_features])
    X_train = np.hstack([X_train_num, X_train_cat])
    X_test = np.hstack([X_test_num, X_test_cat])
    print("Training samples:", X_train.shape[0])
    print("Testing samples:", X_test.shape[0])
    print("Processed features:", X_train.shape[1])
    model = RegularizedLogisticRegression(X_train, y_train, reg_strength=0.01)
    optimizer = GradientDescent(step_size=0.05, max_iterations=50000)
    weights, bias = optimizer.optimize(model)
    print("\nTraining completed.")
    print("Final bias:", bias)
    print("Number of learned weights:", len(weights))
    final_train_loss = model.compute_loss(weights, bias)
    print("Final training loss:", final_train_loss)
    y_pred = predict(X_test, weights, bias)
    test_accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy = {test_accuracy:.4f}")
