import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(30, 1, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        return x

def model_train(model, dataloader, old_parameters, ntier, optim, loss_fn, lam_reg):

    for _ in range(ntier):
        inputs, labels = next(iter(dataloader))
        optim.zero_grad()
        predict = model(inputs)
        loss = loss_fn(predict, labels)
        loss_fn_2 = nn.MSELoss(reduction='sum')

        loss_w = [loss_fn_2(w, w_meta) for w, w_meta in zip(model.parameters(), old_parameters)]
        loss_w = torch.sum(torch.stack(loss_w))

        loss += lam_reg*loss_w
        
        loss.backward()
        optim.step()


def train_loss(model, dataloader, old_parameters, loss_fn, lam_reg):
    loss = 0
    loss_w = 0
    idx = 0
    for ipt, label in dataloader:
        loss += loss_fn(model(ipt), label).item()
        idx += 1
    loss_fn_2 = nn.MSELoss(reduction='sum')
    loss_w = [loss_fn_2(w, w_meta) for w, w_meta in zip(model.parameters(), old_parameters)]
    loss_w = torch.sum(torch.stack(loss_w))
    loss_w = lam_reg*(loss_w.item())
    return (loss/idx + loss_w)