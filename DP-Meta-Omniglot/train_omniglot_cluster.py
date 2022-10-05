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
parser = argparse.ArgumentParser('Train reptile on omniglot')

# Mode
parser.add_argument('logdir', help='Folder to store everything/load')

# - Training params
parser.add_argument('--classes', default=5, type=int, help='classes in base-task (N-way)')
parser.add_argument('--shots', default=5, type=int, help='shots per class (K-shot)')
parser.add_argument('--train-shots', default=10, type=int, help='train shots')
parser.add_argument('--meta-iterations', default=100000, type=int, help='number of meta iterations')
parser.add_argument('--start-meta-iteration', default=0, type=int, help='start iteration')
parser.add_argument('--iterations', default=5, type=int, help='number of base iterations')
parser.add_argument('--test-iterations', default=50, type=int, help='number of base iterations')
parser.add_argument('--batch', default=10, type=int, help='minibatch size in base task')
parser.add_argument('--meta-lr', default=1., type=float, help='meta learning rate')
parser.add_argument('--lr', default=1e-3, type=float, help='base learning rate')
parser.add_argument('--meta-batch', default=25, type=int, help='meta batch size')
parser.add_argument('--noise-multiplier', default=0.423, type=float, help='multiplier')
parser.add_argument('--q', default=2, type=int, help='number of meta models')

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

# By default, continue training
# Check if args.json exists
if os.path.exists(args_filename):
    print('Attempting to resume training. (Delete {} to start over)'.format(args.logdir))
    # Resuming training is incompatible with other checkpoint
    # than the last one in logdir
    assert args.checkpoint == '', 'Cannot load other checkpoint when resuming training.'
    # Attempt to find checkpoint in logdir
    args.checkpoint = args.logdir
else:
    print('No previous training found. Starting fresh.')
    # Otherwise, initialize folders
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)
    if not os.path.exists(check_dir):
        os.makedirs(check_dir)
    # Write args to args.json
    with open(args_filename, 'wb') as fp:
        json.dump(vars(bytes(args)), fp, indent=4)


# Create tensorboard logger
logger = SummaryWriter(run_dir)

# Load data
# Resize is done by the MetaDataset because the result can be easily cached
omniglot = MetaOmniglotFolder('omniglot', size=(28, 28), cache=ImageCache(),
                              transform_image=transform_image,
                              transform_label=transform_label)

triple_minist = omniglot

meta_train_ogt, meta_test_ogt = split_omniglot(omniglot, args.validation)
meta_train_tm, meta_test_tm = split_omniglot(triple_minist, args.validation)

meta_train = [meta_train_ogt, meta_train_ogt]
meta_test = [meta_test_ogt,  meta_test_ogt]


#-----------Generate the dataset-------------------------------------------------------------------------------------------
training_dataset = []
for i in range(50000):
    choice = np.random.random()
    if choice <= 0.5:
        dataset_idx = 0
    else:
        dataset_idx = 1
    character_indices = np.random.choice(len(meta_train[dataset_idx]), args.classes, replace=False)
    img_indices = []
    for i in range(len(character_indices)):
        img_indices.append(np.random.choice(20, args.train_shots + 1, replace=False))
    training_dataset.append((dataset_idx, character_indices, img_indices))
#---------------------------------------------------------------------------------------------------------




# Loss
cross_entropy = nn.NLLLoss()
def get_loss(prediction, labels):
    return cross_entropy(prediction, labels)


def do_learning(net, optimizer, train_iter, iterations, meta_learner_parameters):

    net.train()
    for iteration in range(iterations):
        # Sample minibatch
        data, labels = Variable_(next(train_iter))

        # Forward pass
        prediction = net(data)

        # Get loss
        loss = get_loss(prediction, labels)
        loss_mse = nn.MSELoss(reduction='sum')
        loss_w = [loss_mse(w, w_meta) for w, w_meta in zip(net.parameters(), meta_learner_parameters)]
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
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, betas=(0, 0.999))
    #optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
    if state is not None:
        optimizer.load_state_dict(state)
    return optimizer


def set_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# Build model, optimizer, and set states
q = args.q
meta_nets = []
for _ in range(q):
    meta_nets.append(OmniglotModel(args.classes))
if args.cuda:
    for meta_net in meta_nets:
        meta_net.cuda()

meta_optimizers =[]
for meta_net in meta_nets:
    meta_optimizers.append(torch.optim.SGD(meta_net.parameters(), lr=args.meta_lr))
info = {}
state = None


number_of_first_model = 0
number_of_second_model = 0
# Main loop
for meta_iteration in tqdm.trange(args.start_meta_iteration, args.meta_iterations):

    # Update learning rate
    meta_lr = args.meta_lr * (1. - meta_iteration/float(args.meta_iterations))
    for meta_optimizer in meta_optimizers:
        set_learning_rate(meta_optimizer, meta_lr)

    # Clone model
    gradient_sums = []
    for i in range(q):
        gradient_sums.append([])
    for i in range(args.meta_batch):
        task_idx = np.random.choice(len(training_dataset), 1)[0]
        dataset_idx, character_indices, img_indices = training_dataset[task_idx]
        train, _ = meta_train[dataset_idx].get_task_split(character_indices, img_indices, args.train_shots, 1)
        train_iter = make_infinite(DataLoader(train, args.batch, shuffle=True))
        losses = []
        best_loss = np.inf
        for i, meta_net in enumerate(meta_nets):
            net = meta_net.clone()
            optimizer = get_optimizer(net)


            # Update fast net
            meta_paras = [copy.deepcopy(para) for para in meta_net.parameters()]
            loss = do_learning(net, optimizer, train_iter, args.iterations, meta_paras)
            state = optimizer.state_dict()  # save optimizer state
            if loss < best_loss:
                best_loss = loss
                best_para = [para*2*0.5 for para in net.parameters()]
                best_idx = i
        if best_idx == 0:
            number_of_first_model += 1
        else:
            number_of_second_model += 1
        gradient = var_substract(meta_net.parameters(), best_para)
        gradient = gradient_clipping(gradient, 0.5)
    
        gradient_sums[best_idx] = var_add(gradient_sums[best_idx], gradient)
    
    for i in range(len(gradient_sums)):
        if not gradient_sums[i]:
            gradient_sums[i] = [torch.zeros_like(para) for para in net.parameters()]
        else:
            gradient_sums[i] = var_scale(gradient_sums[i], 1.0/args.meta_batch)

    # Update slow net
    noise_std = 0.5 * args.noise_multiplier/args.meta_batch
    for meta_net, gradient_sum, meta_optimizer in zip(meta_nets, gradient_sums, meta_optimizers):
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
                choice = np.random.random()
                if choice<=0.5:
                    dataset_idx = 0
                else:
                    dataset_idx = 1
                train, test = meta_dataset[dataset_idx].get_random_task_split(args.classes, train_K=args.shots, test_K=1)  # is that 5 ok?
                train_iter = make_infinite(DataLoader(train, args.batch, shuffle=True))
                test_iter = make_infinite(DataLoader(test, args.classes, shuffle=True))
                best_loss = np.inf
                best_net = []
                for meta_net in meta_nets:
                    net = meta_net.clone()
                    optimizer = get_optimizer(net)  # do not save state of optimizer
                    meta_paras = [copy.deepcopy(para) for para in meta_net.parameters()]
                    loss = do_learning(net, optimizer, train_iter, args.test_iterations, meta_paras)
                    if loss < best_loss:
                        best_net = net
                        best_loss = loss

                # Base-test: compute meta-loss, which is base-validation error
                meta_loss, meta_accuracy = do_evaluation(best_net, test_iter, 1)  # only one iteration for eval
                average_acc += meta_accuracy
                average_accs.append(meta_accuracy)
                average_loss += meta_loss

            average_acc /= 400
            average_loss /= 400
            std = np.std(average_accs, 0)
            ci95 = 1.96*std/np.sqrt(400)
            # (Logging)
            loss_ = '{}_loss'.format(mode)
            accuracy_ = '{}_accuracy'.format(mode)
            ci95_ ='{}_ci95'.format(mode)
            meta_lr_ = 'meta_lr'
            info.setdefault(loss_, {})
            info.setdefault(accuracy_, {})
            info.setdefault(meta_lr_, {})
            info.setdefault(ci95_, {})
            info[loss_][meta_iteration] = average_loss
            info[accuracy_][meta_iteration] = average_acc
            info[meta_lr_][meta_iteration] = meta_lr
            info[ci95_][meta_iteration] = ci95
            print('\nMeta-{}'.format(mode))

            print('metaloss', average_loss)
            print('accuracy', average_acc)
            print('ci95', ci95)
            print('number of first model', number_of_first_model)
            print('number_of_second_model', number_of_second_model)
            logger.add_scalar(loss_, meta_loss, meta_iteration)
            logger.add_scalar(accuracy_, meta_accuracy, meta_iteration)
            logger.add_scalar(meta_lr_, meta_lr, meta_iteration)