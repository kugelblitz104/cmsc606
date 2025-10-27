import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def my_train(train_dataset):
    """
    Train a model on the given dataset.

    Args:
        train_dataset:
            The dataset to train on.

    Returns:
        Tuple[W, b]:
            The trained weight matrix, shape of (num_classes, num_features)
            The trained bias vector, shape of (num_classes,)
    """
    # hyperparameters
    learning_rate = 0.1
    l1_lambda = 0.005
    num_epochs = 5
    batch_size = 64

    # create data loader for fast image processing & batching
    return train_with_params(
        train_dataset, learning_rate, l1_lambda, num_epochs, batch_size
    )


def train_with_params(train_dataset, learning_rate, l1_lambda, num_epochs, batch_size):
    # hyperparameters
    num_features = 784  # 28x28 pixels
    num_classes = 10
    learning_rate = learning_rate
    l1_lambda = l1_lambda
    num_epochs = num_epochs
    batch_size = batch_size

    # create data loader for fast image processing & batching
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # initialize weights and bias
    W = torch.randn(num_classes, num_features, requires_grad=True)
    b = torch.zeros(num_classes, requires_grad=True)

    # using cross-entropy loss
    criterion = nn.CrossEntropyLoss()

    # training
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for images, labels in train_loader:
            # flatten images: (batch_size, 1, 28, 28) -> (batch_size, 784)
            images = images.view(images.size(0), -1)

            # forward pass: compute logits
            logits = torch.matmul(images, W.t()) + b

            # compute cross-entropy loss
            ce_loss = criterion(logits, labels)

            # l1 regularization
            ## compute l1 regularization term
            l1_reg = l1_lambda * torch.sum(torch.abs(W))

            ## total loss
            loss = ce_loss + l1_reg

            # backwards propagation
            loss.backward()

            # perform gradient descent update
            with torch.no_grad():
                W -= learning_rate * W.grad  # type: ignore
                b -= learning_rate * b.grad  # type: ignore

                # Zero gradients
                W.grad.zero_()  # type: ignore
                b.grad.zero_()  # type: ignore

            epoch_loss += loss.item()

        print(
            f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss / len(train_loader):.4f}"
        )

    # convert tensors to numpy arrays
    W_np = W.detach().numpy()
    b_np = b.detach().numpy()

    return W_np, b_np


def my_test(W, b, test_dataset):
    """
    Test a model on given dataset

    Args:
        W (np.ndarray): The weight matrix.
        b (np.ndarray): The bias vector.
        test_dataset (np.ndarray): The dataset to test on.

    Returns:
        test_error_rate(float): The accuracy of the model on the test dataset.
    """
    # convert numpy arrays back to torch tensors
    W_torch = torch.from_numpy(W).float()
    b_torch = torch.from_numpy(b).float()

    # create data loader for fast image processing & batching
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            # flatten images: (batch_size, 1, 28, 28) -> (batch_size, 784)
            images = images.view(images.size(0), -1)

            # forward pass: compute logits
            logits = torch.matmul(images, W_torch.t()) + b_torch

            # get predictions (class with highest logit)
            _, predicted = torch.max(logits, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # calculate accuracy
    accuracy = correct / total
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    return accuracy
