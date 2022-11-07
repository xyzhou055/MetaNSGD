import os
import argparse
import tqdm
import json
import re
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from tensorboardX import SummaryWriter
import copy

from models import OmniglotModel
from omniglot import MetaOmniglotFolder, split_omniglot, ImageCache, transform_image, transform_label
from utils import find_latest_file, var_add, var_scale, var_substract, gradient_clipping


def make_infinite(dataloader):
    while True:
        for x in dataloader:
            yield x

def Variable_(tensor, *args_, **kwargs):
    '''
    Make variable cuda depending on the arguments
    '''
    # Unroll list or tuple
    if type(tensor) in (list, tuple):
        return [Variable_(t, *args_, **kwargs) for t in tensor]
    # Unroll dictionary
    if isinstance(tensor, dict):
        return {key: Variable_(v, *args_, **kwargs) for key, v in list(tensor.items())}
    # Normal tensor
    variable = Variable(tensor, *args_, **kwargs)
    if args.cuda:
        variable = variable.cuda()
    return variable

# Parsing
parser = argparse.ArgumentParser('Train reptile onomniglot')

# Mode
parser.add_argument('logdir', help='Folder to store everything/load')

# - Training params
parser.add_argument('--classes', default=5, type=int, help='classes in base-task (N-way)')
parser.add_argument('--shots', default=5, type=int, help='shots per class (K-shot)')
parser.add_argument('--train-shots', default=10, type=int, help='train shots')
parser.add_argument('--meta-iterations', default=100000, type=int, help='number of meta iterations')
parser.add_argument('--start-meta-iteration', default=0, type=int, help='start iteration')
parser.add_argument('--iterations', default=8, type=int, help='number of base iterations')
parser.add_argument('--test-iterations', default=50, type=int, help='number of base iterations')
parser.add_argument('--batch', default=10, type=int, help='minibatch size in base task')
parser.add_argument('--meta-lr', default=1., type=float, help='meta learning rate')
parser.add_argument('--lr', default=1e-3, type=float, help='base learning rate')
parser.add_argument('--meta-batch', default=25, type=int, help='meta batch size')
parser.add_argument('--noise-multiplier', default=0.423, type=float, help='multiplier')

# - General params
parser.add_argument('--validation', default=0.1, type=float, help='Percentage of validation')
parser.add_argument('--validate-every', default=100, type=int, help='Meta-evaluation every ... base-tasks')
parser.add_argument('--input', default='omniglot', help='Path to omniglot dataset')
parser.add_argument('--cuda', default=1, type=int, help='Use cuda')
parser.add_argument('--check-every', default=10000, type=int, help='Checkpoint every')
parser.add_argument('--checkpoint', default='', help='Path to checkpoint. This works only if starting fresh (i.e., no checkpoints in logdir)')

# Do some processing
args = parser.parse_args()
print(args)
args_filename = os.path.join(args.logdir, 'args.json')
run_dir = args.logdir
check_dir = os.path.join(run_dir, 'checkpoint')


# Load data
# Resize is done by the MetaDataset because the result can be easily cached
omniglot = MetaOmniglotFolder(args.input, size=(28, 28), cache=ImageCache(),
                              transform_image=transform_image,
                              transform_label=transform_label)
meta_train, meta_test = split_omniglot(omniglot, args.validation)

# character_indices = np.random.choice(len(meta_train), args.classes, replace=False)
# img_indices = []
# for i in range(len(character_indices)):
#     img_indices.append(np.random.choice(20, args.train_shots + 1))

# train_data, _ = meta_train.get_task_split(character_indices, img_indices, args.train_shots, 1)

#-----------Generate the dataset-------------------------------------------------------------------------------------------
training_dataset = []
for i in range(50000):
    character_indices = np.random.choice(len(meta_train), args.classes, replace=False)
    img_indices = []
    for i in range(len(character_indices)):
        img_indices.append(np.random.choice(20, args.train_shots + 1, replace=False))
    training_dataset.append((character_indices, img_indices))
#---------------------------------------------------------------------------------------------------------


# Loss
cross_entropy = nn.NLLLoss()
def get_loss(prediction, labels):
    return cross_entropy(prediction, labels)


def do_learning(net, optimizer, train_iter, iterations, meta_paras):

    net.train()
    for iteration in range(iterations):
        # Sample minibatch
        data, labels = Variable_(next(train_iter))

        # Forward pass
        prediction = net(data)

        # Get loss
        loss = get_loss(prediction, labels)
        loss_mse = nn.MSELoss(reduction='sum')
        loss_w = [loss_mse(w, w_meta) for w, w_meta in zip(net.parameters(), meta_paras)]
        loss_w = torch.sum(torch.stack(loss_w))
        loss = loss + 0.1 * loss_w

        # Backward pass - Update fast net
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss.data.item()


def do_evaluation(net, test_iter, iterations):

    losses = []
    accuracies = []
    net.eval()
    for iteration in range(iterations):
        # Sample minibatch
        data, labels = Variable_(next(test_iter))

        # Forward pass
        prediction = net(data)

        # Get loss
        loss = get_loss(prediction, labels)

        # Get accuracy
        argmax = net.predict(prediction)
        accuracy = (argmax == labels).float().mean()

        losses.append(loss.data.item())
        accuracies.append(accuracy.data.item())

    return np.mean(losses), np.mean(accuracies)


def get_optimizer(net, state=None):
    #optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, betas=(0, 0.999))
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
    if state is not None:
        optimizer.load_state_dict(state)
    return optimizer


def set_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# Build model, optimizer, and set states
meta_net = OmniglotModel(args.classes)
if args.cuda:
    meta_net.cuda()
meta_optimizer = torch.optim.SGD(meta_net.parameters(), lr=args.meta_lr)
net = meta_net.clone()
#info = {}
state = None

net = meta_net.clone()
optimizer = get_optimizer(net)


# Main loop
acc_list=  []
loss_list = []
for meta_iteration in tqdm.trange(args.start_meta_iteration, args.meta_iterations):

    # Update learning rate
    meta_lr = args.meta_lr * (1. - meta_iteration/float(args.meta_iterations))
    set_learning_rate(meta_optimizer, meta_lr)

    
    # Clone model
    gradient_sum = []
    for i in range(args.meta_batch):
        net.load_state_dict(meta_net.state_dict())
        
        task_idx = np.random.choice(len(training_dataset), 1)[0]
        character_indices, img_indices = training_dataset[task_idx]
        train, _ = meta_train.get_task_split(character_indices, img_indices, args.train_shots, 1)
        train_iter = make_infinite(DataLoader(train, args.batch, shuffle=True))

        # Update fast net
        meta_paras = [copy.deepcopy(para) for para in meta_net.parameters()]
        loss = do_learning(net, optimizer, train_iter, args.iterations, meta_paras)

        gradient = var_substract(meta_net.parameters(), net.parameters())
        gradient = gradient_clipping(gradient, 0.5)

        gradient_sum = var_add(gradient_sum, gradient)
    gradient_sum = var_scale(gradient_sum, 1.0/args.meta_batch)
   
    # Update slow net
    noise_std = 0.5 * args.noise_multiplier/args.meta_batch
   
    meta_net.point_grad_to(gradient_sum, noise_std)
    meta_optimizer.step()

    

    # Meta-Evaluation
    if (meta_iteration+1) % args.validate_every == 0:
        print('\n\nMeta-iteration', meta_iteration)
        print('(started at {})'.format(args.start_meta_iteration))
        print('Meta LR', meta_lr)

        for (meta_dataset, mode) in [(meta_train, 'train'), (meta_test, 'val')]:
            # Base-train
            average_acc = 0
            average_loss = 0
            average_accs = []
            for i in range(400):
                train, test = meta_dataset.get_random_task_split(args.classes, train_K=args.shots, test_K=1)  
                train_iter = make_infinite(DataLoader(train, args.batch, shuffle=True))
                test_iter = make_infinite(DataLoader(test, args.classes, shuffle=True))
                net.load_state_dict(meta_net.state_dict())
                meta_paras = [copy.deepcopy(para) for para in meta_net.parameters()]
                loss = do_learning(net, optimizer, train_iter, args.test_iterations, meta_paras)

                # Base-test: compute meta-loss, which is base-validation error
                meta_loss, meta_accuracy = do_evaluation(net, test_iter, 1)  # only one iteration for eval
                average_acc += meta_accuracy
                average_accs.append(meta_accuracy)
                average_loss += meta_loss

            average_acc /= 400
            average_loss /= 400
            std = np.std(average_accs, 0)
            ci95 = 1.96*std/np.sqrt(400)
            acc_list.append(average_acc)
            loss_list.append(average_loss)
            np.save("acc.npy", np.array(acc_list))
            np.save("loss.npy", np.array(loss_list))
            print('\nMeta-{}'.format(mode))

            print('metaloss', average_loss)
            print('accuracy', average_acc)
            print('ci95', ci95)
            


