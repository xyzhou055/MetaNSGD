from torch.autograd import grad
from torch.nn import parameter
import torch
from torch.utils.data import  DataLoader

from torch import Tensor
import numpy as np
from util import task_sampling, meta_update, gradient_clipping, Add_noise, average_vars_batch
import copy
from numpy.linalg import norm
from torch.utils.data import DataLoader
from model import model_train, train_loss




class MetaDPSGD():

    def __init__(self, models, q):
        self.models = models
        self.keys = list(models[0].state_dict().keys())
        self.q = q

    def train_step(self, N_train_task,  inner_iters,
                   meta_step_size, meta_batch_size,
                lam_reg, optimizer_list, loss, maximum_norm, noise_multiplier, flag):

        """
        Perform one training step of meta DP-SGD
        :param base_learner: the model of base_learner
        :param datasets: the data set
        :param inner_iters: number of inner-loop iterations
        :param meta_step_size: step size for meta algorithm
        :param meta_batch_size: number of tasks sampled in each iteration
        :param sigma: std of added noise to preserve privacy
        :param lam_reg: regularization parameter
        :return:
        """
        train_path = './data/train/'
       

        new_vars = [[] for _ in range(self.q)]
        
        task_index = task_sampling(N_train_task, meta_batch_size)

        
        old_state_dict_list = [copy.deepcopy(self.models[i].state_dict()) for i in range(self.q)]
        old_parameters_list = [[copy.deepcopy(para) for para in self.models[i].parameters()] for i in range(self.q)]
        for task in task_index:
            best_model_index = -1
            best_parameters = []
            best_training_loss = np.inf
            train_data = torch.load(train_path+str(task)+'.pt', map_location='cuda:0')
            train_dataloader = DataLoader(train_data, batch_size=4, shuffle=True)
            for i in range(self.q):
                model = self.models[i]
                optim = optimizer_list[i]
                model_train(model, train_dataloader, old_parameters_list[i], inner_iters, optim, loss, lam_reg)
                training_loss = train_loss(model, train_dataloader, old_parameters_list[i], loss, lam_reg)
                if training_loss < best_training_loss:
                    best_model_index = i
                    best_training_loss = training_loss
                    best_parameters = [copy.deepcopy(para) for para in model.parameters()]
                model.load_state_dict(old_state_dict_list[i])

            gradient = [-lam_reg*(v - val) for v, val in zip(best_parameters, old_parameters_list[best_model_index])]
            #print(gradient)
            if flag == True:
                gradient = gradient_clipping(gradient, maximum_norm)
            new_vars[best_model_index].append(gradient)
            
        
        for i in range(self.q):
            if new_vars[i]:
                #gradient = average_vars(new_vars[i])
                gradient= average_vars_batch(new_vars[i], meta_batch_size)
                #print(gradient)

                #gradient = Add_noise(gradient, noise_multiplier, maximum_norm, meta_batch_size)
                new_states = meta_update(old_parameters_list[i], gradient, meta_step_size)
            else:
                new_states = copy.deepcopy(old_parameters_list[i])

            if flag == True:
                new_states = Add_noise(new_states, noise_multiplier, maximum_norm, meta_batch_size, meta_step_size)
            state_dict = {}

            for key, val in zip(self.keys, new_states):
                state_dict[key] = val
            self.models[i].load_state_dict(state_dict)
                
        #print(self.model.state_dict())

    def evaluate(self,  inner_iters, optimizer_list, loss, lam_reg):
        test_path = './data/test/'
        transfer_risk = []
        old_state_dict_list = [copy.deepcopy(self.models[i].state_dict()) for i in range(self.q)]
        old_parameters_list = [[copy.deepcopy(para) for para in self.models[i].parameters()] for i in range(self.q)]
        for j in range(400):
            idx = j+1
            test_data = torch.load(test_path+str(idx)+'.pt', map_location='cuda:0')
            train_set, test_set = torch.utils.data.random_split(test_data, [10, 40])
            train_loader = DataLoader(train_set, batch_size=4, shuffle=True)

            best_model_index = -1
            best_training_loss = np.inf
            best_parameters = []

            for i in range(self.q):
                model = self.models[i]
                optim = optimizer_list[i]
                model_train(model, train_loader, old_parameters_list[i], inner_iters, optim, loss, lam_reg)
                training_loss = train_loss(model, train_loader, old_parameters_list[i], loss, lam_reg)
                if training_loss < best_training_loss:
                    best_model_index = i
                    best_training_loss = training_loss
            

            model = self.models[best_model_index]
            running_loss = 0
            test_loader = DataLoader(test_set, batch_size=1, shuffle=True)
            for ipt, label in test_loader:
                running_loss += loss(model(ipt), label).item()
            transfer_risk.append(running_loss/40)
            for i in range(self.q):
                self.models[i].load_state_dict(old_state_dict_list[i])

        print(np.average(transfer_risk))
        return np.average(transfer_risk)