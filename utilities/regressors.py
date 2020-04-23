import numpy as np


class LinearRegressor(object):
    """Linear Regressor object.

    A linear regressor object has a training set given by X, y and a vector w.
    It computes the response as y_hat = w @ x

    Parameters
    ----------
    x: np.ndarray.
        Co-variates vector.
    y: np.ndarray.
        Vector of responses.
    """
    _eps: float = 1e-12  # numerical precision

    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        self._Xtr = x
        self._Ytr = y
        self.weights = np.random.randn(x.shape[1])

    @property
    def number_samples(self) -> int:
        """Return length of training set."""
        return self._Xtr.shape[0]

    def loss(self, w: np.ndarray, indexes: np.ndarray = None) -> np.ndarray:
        """Get loss of w and the current index."""
        if indexes is None:
            indexes = np.arange(self.number_samples)

        self.weights = w
        error = self.predict(self._Xtr[indexes, :]) - self._Ytr[indexes]
        return np.dot(error.T, error) / indexes.size

    def gradient(self, w: np.ndarray, indexes: np.ndarray = None) -> np.ndarray:
        """Get gradient of w and the current index."""
        if indexes is None:
            indexes = np.arange(self.number_samples)

        self.weights = w
        error = self.predict(self._Xtr[indexes, :]) - self._Ytr[indexes]
        return np.dot(self._Xtr[indexes, :].T, error) / indexes.size

    def load_data(self, x: np.ndarray, y: np.ndarray):
        """Load training data."""
        self._Xtr = x
        self._Ytr = y

    def close_form_weights(self) -> None:
        """Calculate the weights using the closed form expression."""
        dim = self._Xtr.shape[1]
        self.weights = np.dot(
            np.linalg.pinv(np.dot(self._Xtr.T, self._Xtr) + self._eps * np.eye(dim)),
            np.dot(self._Xtr.T, self._Ytr))

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict an output given the inptus."""
        return np.dot(x, self.weights)

    def test_loss(self, w: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Calculate the test loss with a different w."""
        w_old = self.weights
        self.weights = w
        error = self.predict(x) - y

        self.weights = w_old
        return np.dot(error.T, error)
