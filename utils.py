"""
Utilization functions
"""

import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def str_to_bool(val):
    """Convert a string representation of truth to true (1) or false (0).
    Copied from the python implementation distutils.utils.strtobool

    True values are 'y', 'yes', 't', 'true', 'on', and '1'; false values
    are 'n', 'no', 'f', 'false', 'off', and '0'.  Raises ValueError if
    'val' is anything else.
    >>> str_to_bool('YES')
    1
    >>> str_to_bool('FALSE')
    0
    """
    val = val.lower()
    if val in ('y', 'yes', 't', 'true', 'on', '1'):
        return 1
    elif val in ('n', 'no', 'f', 'false', 'off', '0'):
        return 0
    else:
        raise ValueError("invalid truth value %r" % (val))


def cosine_annealing(step, total_steps, lr_max, lr_min):
    """Cosine Annealing for learning rate decay scheduler"""
    return lr_min + (lr_max -
                     lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))


def keras_decay(step, decay=0.0001):
    """Learning rate decay in Keras-style"""
    return 1. / (1. + decay * step)


class SGDRScheduler(torch.optim.lr_scheduler._LRScheduler):
    """SGD with restarts scheduler"""
    def __init__(self, optimizer, T0, T_mul, eta_min, last_epoch=-1):
        self.Ti = T0
        self.T_mul = T_mul
        self.eta_min = eta_min

        self.last_restart = 0

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        T_cur = self.last_epoch - self.last_restart
        if T_cur >= self.Ti:
            self.last_restart = self.last_epoch
            self.Ti = self.Ti * self.T_mul
            T_cur = 0

        return [
            self.eta_min + (base_lr - self.eta_min) *
            (1 + np.cos(np.pi * T_cur / self.Ti)) / 2
            for base_lr in self.base_lrs
        ]


def _get_optimizer(model_parameters, optim_config):
    """Defines optimizer according to the given config"""
    optimizer_name = optim_config['optimizer']

    if optimizer_name == 'sgd':
        optimizer = torch.optim.SGD(model_parameters,
                                    lr=optim_config['base_lr'],
                                    momentum=optim_config['momentum'],
                                    weight_decay=optim_config['weight_decay'],
                                    nesterov=optim_config['nesterov'])
    elif optimizer_name == 'adam':
        optimizer = torch.optim.Adam(model_parameters,
                                     lr=optim_config['base_lr'],
                                     betas=optim_config['betas'],
                                     weight_decay=optim_config['weight_decay'],
                                     amsgrad=str_to_bool(
                                         optim_config['amsgrad']))
    else:
        print('Un-known optimizer', optimizer_name)
        sys.exit()

    return optimizer


def _get_scheduler(optimizer, optim_config):
    """
    Defines learning rate scheduler according to the given config
    """
    if optim_config['scheduler'] == 'multistep':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=optim_config['milestones'],
            gamma=optim_config['lr_decay'])

    elif optim_config['scheduler'] == 'sgdr':
        scheduler = SGDRScheduler(optimizer, optim_config['T0'],
                                  optim_config['Tmult'],
                                  optim_config['lr_min'])

    elif optim_config['scheduler'] == 'cosine':
        total_steps = optim_config['epochs'] * \
            optim_config['steps_per_epoch']

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: cosine_annealing(
                step,
                total_steps,
                1,  # since lr_lambda computes multiplicative factor
                optim_config['lr_min'] / optim_config['base_lr']))

    elif optim_config['scheduler'] == 'keras_decay':
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda step: keras_decay(step))
    else:
        scheduler = None
    return scheduler


def create_optimizer(model_parameters, optim_config):
    """Defines an optimizer and a scheduler"""
    optimizer = _get_optimizer(model_parameters, optim_config)
    scheduler = _get_scheduler(optimizer, optim_config)
    return optimizer, scheduler


def seed_worker(worker_id):
    """
    Used in generating seed for the worker of torch.utils.data.Dataloader
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def set_seed(seed, config = None):
    """ 
    set initial seed for reproduction
    """
    if config is None:
        raise ValueError("config should not be None")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = str_to_bool(config["cudnn_deterministic_toggle"])
        torch.backends.cudnn.benchmark = str_to_bool(config["cudnn_benchmark_toggle"])


class AMSoftmaxLoss(nn.Module):
    """AM-Softmax loss with dynamic margin based on utterance duration.
    
    Args:
        scale (float): Scale factor (s in the paper). Default: 15.
        adaptive_margin (bool): Whether to use adaptive margin based on duration. Default: False.
        m_a (float): Slope for margin calculation (A in the paper). Default: 3/50.
        m_b (float): Intercept for margin calculation (B in the paper). Default: 7/50.
        m (float): Fixed margin when adaptive_margin is False. Default: 0.2.
    """

    def __init__(self, scale=15.0, adaptive_margin=False, m_a=3/50, m_b=7/50, m=0.2):
        super(AMSoftmaxLoss, self).__init__()
        self.scale = scale  # scale factor (s)
        self.adaptive_margin = adaptive_margin
        self.m_a = m_a  # slope for adaptive margin
        self.m_b = m_b  # intercept for adaptive margin
        self.m = m  # fixed margin

    def forward(self, outputs, targets, durations=None):
        """
        Args:
            outputs (torch.Tensor): Cosine similarity matrix (batch_size, num_classes).
            targets (torch.Tensor): Target classes (batch_size).
            durations (torch.Tensor, optional): Utterance durations in seconds. Required if adaptive_margin is True.
        """
        batch_size = outputs.size(0)
        
        # Calculate margins for each sample if using adaptive margin
        if self.adaptive_margin:
            if durations is None:
                raise ValueError("Durations must be provided when using adaptive margin")
            # Calculate margin based on duration: m = A * duration + B
            margins = self.m_a * durations + self.m_b
        else:
            # Use fixed margin for all samples
            margins = torch.ones_like(targets, dtype=torch.float) * self.m
        
        # Convert margins to device of outputs
        margins = margins.to(outputs.device)
        
        # Apply margin to target logits
        one_hot = torch.zeros_like(outputs)
        one_hot.scatter_(1, targets.view(-1, 1), 1)
        
        # Subtract margin from target class cosine values
        outputs = outputs - one_hot * margins.view(-1, 1)
        
        # Scale all cosine values
        outputs = outputs * self.scale
        
        # Apply cross-entropy loss
        loss = F.cross_entropy(outputs, targets)
        
        return loss
