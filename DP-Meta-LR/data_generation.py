import torch
from torch.utils.data import TensorDataset

import numpy as np
from numpy.linalg import norm


def data_generation():
    d = 30    #dimension

    w_bar_1 = np.concatenate(([2]*10, [0]*10, [0]*10))
    w_bar_2 = np.concatenate(([0]*10, [-4]*10, [0]*10))
    w_bar_3 = np.concatenate(([0]*10, [0]*10, [6]*10))


    N_train_task = 1000
    N_sample_per_task = 10
    N_test_task = 500
    sigma = 0.5
    noise_sigma = 0.5
    train_path = './data/train/'
    test_path = './data/test/'

    for i in range(N_train_task):

        choice = np.random.rand()
        if choice < 0.33:
            w_bar = w_bar_1
        elif choice <0.67:
            w_bar = w_bar_2
        else:
            w_bar = w_bar_3

        w = np.random.normal(w_bar, sigma).reshape(d, 1)
        X = np.random.random((N_sample_per_task, d))
        l2norm = norm(X, axis=1, ord=2)
        X = X/l2norm[:, None]
        noise = np.random.normal(0, noise_sigma, N_sample_per_task).reshape(N_sample_per_task, 1)
        y = np.matmul(X, w) + noise

        tensor_X = torch.Tensor(X)
        tensor_y = torch.Tensor(y)

        dataset = TensorDataset(tensor_X, tensor_y)
        torch.save(dataset, train_path+str(i+1)+'.pt')

    for i in range(N_test_task):

        choice = np.random.rand()
        if choice < 0.33:
            w_bar = w_bar_1
        elif choice < 0.67:
            w_bar = w_bar_2
        else:
            w_bar = w_bar_3
        w = np.random.normal(w_bar, sigma).reshape(d, 1)
        X = np.random.random((N_sample_per_task*5, d))
        l2norm = norm(X, axis=1, ord=2)
        X = X / l2norm[:, None]
        noise = np.random.normal(0, noise_sigma, N_sample_per_task*5).reshape(N_sample_per_task*5, 1)
        y = np.matmul(X, w) + noise

        tensor_X = torch.Tensor(X)
        tensor_y = torch.Tensor(y)

        dataset = TensorDataset(tensor_X, tensor_y)
        torch.save(dataset, test_path + str(i + 1) + '.pt')