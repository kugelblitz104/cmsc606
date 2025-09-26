import numpy as np
import pandas as pd
from typing import Tuple


def read_csv_convert_to_numpy(fileName: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read a local csv file and convert it to a numpy array

    Args:
        fileName(str): name of a csv file

    Returns:
        numpy_x: feature vectors - # of features x # of samples
        numpy_y: class vector - # of samples x 1
    """
    # read file
    df = pd.read_csv(fileName, index_col=0)

    # convert to numpy array
    numpy_x = df[["ZeroToSixty", "PowerHP"]].to_numpy()
    numpy_y = np.where(df["IsCar"] > 0, 1, -1)
    return numpy_x, numpy_y.reshape(-1, 1)


def calc_error_rate_for_single_vector_w(
    w: np.ndarray, numpy_x: np.ndarray, numpy_y: np.ndarray
) -> float:
    """
    Calculate the error rate for a single weight vector w

    Args:
        w(ndarray): weight vector - # of features x 1
        numpy_x(ndarray): feature vectors - # of features x # of samples
        numpy_y(ndarray): class vectors - # of samples x 1

    Returns:
        error_rate(float): # of errors / # of samples
    """
    # get predicted values
    pred = np.sign(numpy_x @ w)
    # calc error rate
    return np.sum(pred != numpy_y) / numpy_x.shape[0]


def train_and_evaluate(
    numpy_x: np.ndarray, numpy_y: np.ndarray, n_epochs: int = 20, c: float = 0.01
) -> np.ndarray:
    """
    Train random weights based on given data and hyperparameters

    Args:
        numpy_x(ndarray): feature vectors - # of samples x # of features
        numpy_y(ndarray): class vectors - # of samples x 1
        n_epochs(int): number of times to repeat training
        c(float): learning rate, how much of an adjustment to make each time there is an error

    Returns:
        w(ndarray): weight vector - # of features x 1
        finished weights after training
    """
    # generate random weights
    w = np.random.randn(numpy_x.shape[1], 1)
    for epoch in range(n_epochs):
        for i, sample in enumerate(numpy_x):
            pred = np.sign(sample.T @ w)
            true_y = numpy_y[i]
            if pred != true_y:
                w += c * 2.0 * true_y * sample.reshape(-1, 1)

        # calculate error rate
        error_rate = calc_error_rate_for_single_vector_w(w, numpy_x, numpy_y)

        # print error rate for each epoch
        print(error_rate)

        if error_rate == 0:
            break


    return w


def function_error_rate_2D(
    w1_range: np.ndarray,
    w2_range: np.ndarray,
    numpy_x: np.ndarray,
    numpy_y: np.ndarray,
):
    """
    Calculate the error rate for all possible weight vectors
    within given ranges w1_range and w2_range

    Args:
        w1_range(ndarray): range of values for w1
        w2_range(ndarray): range of values for w2
        numpy_x(ndarray): feature vectors - # of samples x # of features
        numpy_y(ndarray): class vectors - # of samples x 1

    Returns:
        error_rates(ndarray): error rates for all possible weight vectors
    """
    # create mesh grids for w1 range and w2 range
    w1, w2 = np.meshgrid(w1_range, w2_range, indexing="ij")

    # stack mesh grids to get all possible w arrays
    w_stack = np.stack([w1, w2], axis=-1)

    # get predictions
    y_pred = np.sign(w_stack @ numpy_x.T).astype("int64")

    # reshape y into a column vector for comparison with y_pred
    y = numpy_y.reshape(1, 1, -1)

    # calculate error rates for all weights
    return np.mean(y_pred != y, axis=2).T
