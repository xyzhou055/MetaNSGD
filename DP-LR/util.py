import numpy as np
import copy
import math
import torch


def task_sampling(n, m):
    """
    Randomly sample m different values from [1,2,...n]
    :param n: size of selection space
    :param m: number of values to be selected
    :return: np array with size m
    """

    return np.random.choice(n,m, replace=False)+1




def average_vars(new_vars):
    """
    Compute the average of the parameters from several base_learners
    :param new_vars: a list of state dictionary of base_learner from the same meta batch
    :return: one state dict which contains the average
    """
    res = []
    for v in zip(*new_vars):
        res.append(torch.mean(torch.stack(v), dim=0))
    return res

def average_vars_batch(new_var, meta_batch_size):
    size = len(new_var)
    res = average_vars(new_var)
    res_batch = []
    for para in res:
        res_batch.append(para*size/meta_batch_size)
    return res_batch

def Add_noise(gradient, noise_multiplier, maximum_norm, meta_batch_size,meta_step_size):
    res = []
    for para in gradient:
        res.append(torch.normal(para, meta_step_size*noise_multiplier*maximum_norm/meta_batch_size))
    return res

def print_gradient(new_vars):
    batch_size = len(new_vars)
    for i in range(batch_size):
        grad_norm = 0
        for key in new_vars[0].keys():
            grad_norm += new_vars[i][key].pow(2).sum()
        grad_norm = math.sqrt(grad_norm)
        print(grad_norm)


def gradient_clipping(gradient, L):
    norm = 0
    for para in gradient:
        norm += torch.sum(para ** 2).item()
    
    norm = np.sqrt(norm)
    if norm > L:
        gradient = [g *(L/norm) for g in gradient]
    return gradient

    


def meta_update(old_paras, gradient, meta_step_size):
    """
    Perform one step of update of the meta model
    :param lam_reg: regularization parameter
    :param meta_step_size: the step size of meta update
    :param old_state: the meta state from last step
    :param base_learner_param: the parameter of the average of base learners
    :return: Updated state
    """

    new_states = [v -meta_step_size*val for v, val in zip(old_paras, gradient)]
    return new_states


def computer_sigma(number_of_tasks, meta_iteration, clipped_norm, epsilon):

    return np.sqrt(8*meta_iteration*np.power(clipped_norm, 2) *
                   np.log(number_of_tasks)/np.power(number_of_tasks*epsilon, 2))
