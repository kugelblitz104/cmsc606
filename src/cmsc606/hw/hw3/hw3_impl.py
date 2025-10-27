import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def my_train(train_dataset):
    """
    Train a model on the given dataset.

    Args:
        train_dataset(MNIST):
            The dataset to train on.

    Returns:
        Tuple[W, b]:
            The trained weight matrix, shape of (num_classes, num_features)
            The trained bias vector, shape of (num_classes,)
    """
    # hyperparameters
    learning_rate = 1.2
    l1_lambda = 0.001
    lr_decay = 0.3
    num_epochs = 5
    batch_size = 256

    return train_with_params(
        train_dataset, learning_rate, lr_decay, l1_lambda, num_epochs, batch_size
    )


def train_with_params(
    train_dataset,
    learning_rate: float,
    lr_decay: float,
    l1_lambda: float,
    num_epochs: int,
    batch_size: int,
):
    """
    Perform training with given hyperparameters

    Args:
        train_dataset (MNIST): training dataset
        learning_rate (float): learning rate
        lr_decay (float): learning rate decay, rate at which to decay learning rate each epoch
        l1_lambda (float): L1 normalization factor
        num_epochs (int): # of epochs
        batch_size (int): batch size

    Returns:
        Tuple[W, b]:
            The trained weight matrix, shape of (num_classes, num_features)
            The trained bias vector, shape of (num_classes,)
    """
    # hyperparameters
    num_features = 784  # 28x28 pixels
    num_classes = 10
    learning_rate = learning_rate
    lr_decay = lr_decay
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
            # compute l1 regularization term
            l1_reg = l1_lambda * torch.sum(torch.abs(W))

            # total loss
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
            f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss / len(train_loader):.4f}, LR: {learning_rate:.4f}"
        )

        # decay learning rate for aggresive initial learning rate
        # that adjusts to be smaller each epoch for smaller adjustments
        learning_rate = learning_rate * (1 - lr_decay)

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

    # calculate accuracy & error rate
    accuracy = correct / total
    error_rate = 1 - accuracy
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    print(f"Test error rate: {error_rate * 100:.2f}%")

    if error_rate < 0.125:
        print("✅ Test passed, error rate is below 12.5%")
    else:
        print("❌ Test failed, error rate is above 12.5%")

    return error_rate
