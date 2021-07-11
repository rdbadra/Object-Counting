from haven import haven_utils as hu
import torch
import torchvision
import tqdm
import pandas as pd
import pprint
import itertools
import os
import pylab as plt
import exp_configs
import time
import numpy as np

from src import models
from src import datasets


import argparse

from torch.utils.data import sampler
from torch.utils.data.sampler import RandomSampler
from torch.backends import cudnn
from torch.nn import functional as F
from torch.utils.data import DataLoader


def train(exp_dict, savedir, datadir, reset=False, num_workers=0):
    os.makedirs(savedir, exist_ok=True)
    # Dataset
    # ==================
    # train set
    score_list = []
    train_set = datasets.get_dataset(dataset_dict=exp_dict["dataset"],
                                     split="train",
                                     datadir=datadir,
                                     exp_dict=exp_dict,
                                     dataset_size=exp_dict['dataset_size'])
    # val set
    val_set = datasets.get_dataset(dataset_dict=exp_dict["dataset"],
                                   split="val",
                                   datadir=datadir,
                                   exp_dict=exp_dict,
                                   dataset_size=exp_dict['dataset_size'])

    val_sampler = torch.utils.data.SequentialSampler(val_set)
    val_loader = DataLoader(val_set,
                            sampler=val_sampler,
                            batch_size=1,
                            num_workers=num_workers)
    # Model
    # ==================
    model = models.get_model(model_dict=exp_dict['model'],
                             exp_dict=exp_dict,
                             train_set=train_set).cuda()

    # model.opt = optimizers.get_optim(exp_dict['opt'], model)
    model_path = os.path.join(savedir, "model.pth")
    score_list_path = os.path.join(savedir, "score_list.pkl")

    # Train & Val
    # ==================
    train_sampler = torch.utils.data.RandomSampler(
        train_set, replacement=True, num_samples=2*len(val_set))

    train_loader = DataLoader(train_set,
                              sampler=train_sampler,
                              batch_size=exp_dict["batch_size"], 
                              drop_last=True, num_workers=num_workers)
    
    best_val = 100000
    epochs_without_improvement = 0
    for e in range(exp_dict['max_epoch']):
        # Validate only at the start of each cycle
        score_dict = {}

        # Train the model
        train_dict = model.train_on_loader(train_loader)
        print(f'train_dict: {train_dict}')

        # Validate and Visualize the model
        val_dict = model.val_on_loader(val_loader, 
                        savedir_images=os.path.join(savedir, "images"),
                        n_images=5, epoch=e)
        print(f'val_dict: {val_dict}')  
        if(val_dict['val_mae'] < best_val):
            print('Better validation')
            best_val = val_dict['val_mae']   
            epochs_without_improvement = 0
        elif(epochs_without_improvement > 3):
            print('No improvement for 4 epochs')
            break
        else:
            epochs_without_improvement += 1
        score_dict.update(val_dict)
        # model.vis_on_loader(
        #     vis_loader, savedir=os.path.join(savedir, "images"))

        # Get new score_dict
        score_dict.update(train_dict)
        score_dict["epoch"] = len(score_list)

        # Add to score_list and save checkpoint
        score_list += [score_dict]

        # Report & Save
        score_df = pd.DataFrame(score_list)
        print("\n", score_df.tail(), "\n")
        '''
        hu.torch_save(model_path, model.get_state_dict())
        hu.save_pkl(score_list_path, score_list)
        '''
        print("Checkpoint Saved: %s" % savedir)

        # Save Best Checkpoint

        if e == 0 or (score_dict.get("val_score", 0) > score_df["val_score"][:-1].fillna(0).max()):
            hu.save_pkl(os.path.join(
                savedir, "score_list_best.pkl"), score_list)
            hu.torch_save(os.path.join(savedir, f"model_best_{e}.pth"),
                          model.get_state_dict())
            print("Saved Best: %s" % savedir)

    score_df.to_csv(os.path.join(savedir, 'score_df.csv'), index=False)

    print('Experiment completed et epoch %d' % e)


if __name__ == "__main__":
    torch.cuda.empty_cache()
    exp_dict = {"dataset":
               {'name': 'trancos',
                'transform': 'rgb_normalize'},
            'model':
               {'name': 'lcfcn',
                'base': 'fcn8_vgg16'},
            'batch_size': 1,
            'max_epoch': 10,
            'dataset_size': 
               {'train': 'all',
                'val': 'all'},
            'optimizer': 'sgd',
            'lr': 1e-5}
    datadir = 'TRANCOS_v3/'

    options = ['adam', 'sgd']
    for option in options:
        exp_dict['optimizer'] = option
        train(
            exp_dict=exp_dict,
            savedir=f'custom_output_{option}',
            datadir=datadir,
            num_workers=1)