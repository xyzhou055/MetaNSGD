import torch
import torch.nn as nn
import torch.optim as optim

from model import Model
from data_generation import data_generation
from MetaDPSGD import MetaDPSGD
from tensorflow_privacy.privacy.analysis.compute_noise_from_budget_lib import compute_noise

device = torch.device('cuda:0')
meta_learners = []
optimizers = []
q = 2

for i in range(q):
    model = Model().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.07)
    meta_learners.append(model)
    optimizers.append(optimizer)

lam_reg = 0.4
inner_iteration = 20
meta_step_size = 0.7

loss_fn = nn.MSELoss()
#-----------------hyperparamters-------------
N_train_task = 50
sampling_rate = 0.05
epoch = 3
maximum_norm = 1
noise_multiplier = 1
#--------------------------------------------------
result = {}
for epsilon in [3,10,10000]:
    result[epsilon] = []
    for N_train_task in range(100, 701, 100):
        print("epsilon:" ,epsilon)
        print("N_train_task", N_train_task)
        running_loss = 0
        for i in range(5): 
            data_generation()
            for j in range(q):
                meta_learner = meta_learners[j]    
                for layer in meta_learner.children():
                    if hasattr(layer, 'reset_parameters'):
                        layer.reset_parameters()
            meta_iteration = int(N_train_task)
            meta_batch_size = 10
            if epsilon > 100:
                noise_multiplier = 0
                maximum_norm = 1
                flag = False
            else:
                noise_multiplier = compute_noise(N_train_task, meta_batch_size, epsilon, meta_batch_size, 1e-5, 1e-6)
                maximum_norm = 1
                flag = True
            meta_SGD = MetaDPSGD(meta_learners, q)
            idx =1
            for _ in range(meta_iteration):
                idx += 1
                if idx%100 == 0:
                    print(idx)
                meta_SGD.train_step(N_train_task, inner_iteration, meta_step_size, meta_batch_size, lam_reg, optimizers, loss_fn, maximum_norm, noise_multiplier, flag)            
            print("evaluation start:..............")
            running_loss += meta_SGD.evaluate(inner_iteration, optimizers, loss_fn, lam_reg)
        
        result[epsilon].append(running_loss/5)
    print(result)
print(result)