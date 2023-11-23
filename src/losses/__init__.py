import numpy as np


def mean_squared_error(y_true, y_pred):
    return np.mean(np.square(y_true - y_pred))


def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def hinge(y_true, y_pred):
    return np.mean(np.maximum(1. - y_true * y_pred, 0.))


def binary_crossentropy(y_true, y_pred):
    return np.mean(-y_true * np.log(y_pred) - (1. - y_true) * np.log(1. - y_pred))


def categorical_crossentropy(y_true, y_pred):
    return np.mean(-np.sum(y_true * np.log(y_pred), axis=-1))


def sparse_categorical_crossentropy(y_true, y_pred):
    return np.mean(-np.sum(y_true * np.log(y_pred), axis=-1))
