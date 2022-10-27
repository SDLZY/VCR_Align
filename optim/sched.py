"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

optimizer learning rate scheduling helpers
"""
import math
from math import ceil


def noam_schedule(step, warmup_step=4000):
    """ original Transformer schedule"""
    if step <= warmup_step:
        return step / warmup_step
    return (warmup_step ** 0.5) * (step ** -0.5)


def warmup_linear(step, warmup_step, tot_step):
    """ BERT schedule """
    if step < warmup_step:
        return step / warmup_step
    return max(0, (tot_step-step)/(tot_step-warmup_step))


def warmup_exp(step, warmup_step, decay_rate):
    """ BERT schedule """
    # import pdb; pdb.set_trace()
    if step < warmup_step:
        return step / warmup_step
    return decay_rate**(step - warmup_step)


def warmup_cos(step, warmup_step, tot_step, lr_max, lr_min):
    """ BERT schedule """
    if step < warmup_step:
        return step / warmup_step
    return (lr_min + 0.5 * (lr_max - lr_min)*(1. + math.cos(math.pi * (step - warmup_step)/(tot_step-warmup_step))))/lr_max


def vqa_schedule(step, warmup_interval, decay_interval,
                 decay_start, decay_rate):
    """ VQA schedule from MCAN """
    if step < warmup_interval:
        return 1/4
    elif step < 2 * warmup_interval:
        return 2/4
    elif step < 3 * warmup_interval:
        return 3/4
    elif step >= decay_start:
        num_decay = ceil((step - decay_start) / decay_interval)
        return decay_rate ** num_decay
    else:
        return 1


def get_lr_sched(global_step, opts):
    # learning rate scheduling
    if opts.lr_decay == 'exp':
        lr_this_step = opts.learning_rate * warmup_exp(global_step, opts.warmup_steps, opts.lr_decay_rate)
    elif opts.lr_decay == 'cos':
        lr_this_step = opts.learning_rate * warmup_cos(global_step, opts.warmup_steps, opts.num_train_steps,
                                                       opts.learning_rate, opts.min_lr)
    elif opts.lr_decay == 'linear':
        lr_this_step = opts.learning_rate * warmup_linear(
            global_step, opts.warmup_steps, opts.num_lr_decay_steps)
        # if lr_this_step <= 0:
        #     lr_this_step = 1e-8

    return max(opts.min_lr, lr_this_step)