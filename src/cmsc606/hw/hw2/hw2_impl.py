import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
    return numpy_x, numpy_y


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
    pred = get_predictions(w, numpy_x)

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
    np.random.seed(3)  # fixed randomness

    # generate random weights
    w = np.random.randn(numpy_x.shape[1])
    for epoch in range(n_epochs):
        pred_y = get_predictions(w, numpy_x)
        error_rate = calc_error_rate_for_single_vector_w(w, numpy_x, numpy_y)

        # print error rate for each epoch
        print(error_rate)

        # calculate loss
        loss = pred_y - numpy_y

        # calculate gradient
        gradient = loss * numpy_x.T

        # calculate weight adjustment
        w_delta = np.sum(gradient, axis=1) * c

        w += w_delta

    return w


# def function_error_rate_2D(
#     w1_range: np.ndarray,
#     w2_range: np.ndarray,
#     numpy_x: np.ndarray,
#     numpy_y: np.ndarray,
# ):
#     """
#     Calculate the error rate for all possible weight vectors
#     within given ranges w1_range and w2_range
#
#     Args:
#         w1_range(ndarray): range of values for w1
#         w2_range(ndarray): range of values for w2
#         numpy_x(ndarray): feature vectors - # of samples x # of features
#         numpy_y(ndarray): class vectors - # of samples x 1
#
#     Returns:
#         error_rates(ndarray): error rates for all possible weight vectors
#     """
#     # create mesh grids for w1 range and w2 range
#     w1, w2 = np.meshgrid(w1_range, w2_range, indexing="ij")
#
#     # stack mesh grids to get all possible w arrays
#     w_stack = np.stack([w1, w2], axis=-1)
#
#     # flatten w_stack to a list of all possible weights
#     # in a 2 column, r1 * r2 len 2d array
#     w_flat = w_stack.reshape(-1, 2)
#
#     # calculate prediction scores
#     scores = numpy_x @ w_flat.T
#
#     # get predictions
#     y_pred = np.sign(scores).astype("int64")
#
#     # reshape y into a column vector for comparison with y_pred
#     y = numpy_y.reshape(-1, 1)
#
#     # calculate error rates for all weights
#     error_rates = np.mean(y_pred != y, axis=0)
#
#     # reshape error_rates to target output size
#     return error_rates.reshape(w1.shape)

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
    error_rates = np.zeros((len(w1_range), len(w2_range)))
    for i, w1 in enumerate(w1_range):
        for j, w2 in enumerate(w2_range):
            error_rate = calc_error_rate_for_single_vector_w(
                np.array([w1, w2]), numpy_x, numpy_y
            )
            error_rates[i, j] = error_rate

    return error_rates


# helper functions
def get_predictions(
    w: np.ndarray,
    numpy_x: np.ndarray,
) -> np.ndarray:
    """
    Get the predictions for a weight vector w and samples x

    Args:
        w(ndarray): weight vector - # of features:
        numpy_x(ndarray): feature vectors - # of samples x # of features:

    Returns:
        predictions(ndarray): class vectors - # of samples
    """
    w = np.squeeze(w)
    return np.sign(np.sum(numpy_x * w, axis=1)).astype("int64")
