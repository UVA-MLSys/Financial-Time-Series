import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math, gc, torch, os
from typing import List
from torch import Tensor
from pandas import DataFrame
from data_provider.data_loader import Dataset_Custom
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from utils.metrics import calculate_metrics

def stringify_setting(args, complete=False):
    if not complete:
        setting = f"{args.model}_sl_{args.seq_len}_pl_{args.pred_len}"
        if args.des:
            setting += '_' + args.des
        
        if 'model_id' in args:
            setting += '_id_' + args.model_id
        if args.percent != 100:
            if args.percent > 0:
                setting += f'_p_{args.percent}'
            else:
                setting += '_zeroshot'
            
        return setting
    
    setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}'.format(
        args.task_name,
        args.model,
        args.data_path.split('.')[0],
        args.data,
        args.features,
        args.seq_len,
        args.label_len,
        args.pred_len,
        args.d_model,
        args.n_heads,
        args.e_layers,
        args.d_layers,
        args.d_ff,
        args.factor,
        args.embed,
        args.distil
    )
    
    return setting

def align_predictions(
    ground_truth:DataFrame,
    predictions, data_set:Dataset_Custom,
    remove_negative:bool=True, upscale:bool=True, 
    disable_progress:bool=False
):
    print('Aligning predictions with ground truth...')
    if type(predictions) != list:
        # shape: N x pred_len x tagets
        if len(predictions.shape) == 3:
            pred_list = [
                predictions[:, :, target] for target in range(predictions.shape[-1])
            ]
        # shape: N x pred_len
        else:
            pred_list = [predictions]
        
        # make it a list of predictions
        predictions = pred_list

    horizons = range(data_set.pred_len)
    predictions_index = data_set.index
    time_index_max = predictions_index[data_set.time_column].max()

    targets, time_index, group_ids = [data_set.target], data_set.time_column, data_set.group_id
    # a groupby with a groupength 1 throws warning later
    # if type(group_ids) == list and len(group_ids) == 1: 
    #     group_ids = group_ids[0]
    
    all_outputs = None 
    for target_index, target in enumerate(targets):
        if type(predictions[target_index]) == Tensor:
            predictions[target_index] = predictions[target_index].numpy()

        pred_df = DataFrame(
            predictions[target_index], columns=horizons
        )
        pred_df = pd.concat([predictions_index, pred_df], axis=1)
        outputs = []

        for group_id, group_df in tqdm(
            pred_df.groupby(group_ids), disable=disable_progress, 
            desc=f'Aligning {target}'
        ):
            group_df = group_df.sort_values(
                by=time_index
            ).reset_index(drop=True)

            new_df = DataFrame({
                time_index : [t + time_index_max for t in range (1, data_set.pred_len)]
            })
            new_df[group_ids] = group_id
            new_df.loc[:, horizons] = None
            new_df = new_df[group_df.columns]
            group_df = pd.concat([group_df, new_df], axis=0).reset_index(drop=True)

            for horizon in horizons:
                group_df[horizon] = group_df[horizon].shift(periods=horizon, axis=0)
                
            group_df[target] = group_df[horizons].mean(axis=1, skipna=True)
            
            # fill the values which are still None
            group_df.fillna(0, inplace=True)
            outputs.append(group_df.drop(columns=horizons))

        outputs = pd.concat(outputs, axis=0)
        
        if all_outputs is None: all_outputs = outputs
        else: 
            all_outputs = all_outputs.merge(
                outputs, how='inner', on=list(predictions_index.columns)
            )
            
    gc.collect()
      
    # upscale the target values if needed
    if upscale:
        all_outputs = data_set.upscale_target(all_outputs)
        
    # must appear after upscaling
    if remove_negative:
        # remove negative values, since infection count can't be negative
        for target in targets:
            all_outputs.loc[all_outputs[target]<0, target] = 0
            
    # add `Predicted` prefix to the predictions
    all_outputs.rename(
        {target:'Predicted_'+target for target in targets}, 
        axis=1, inplace=True
    )

    # only keep the directly relevant columns
    ground_truth = ground_truth[list(predictions_index.columns) + targets]
    
    # merge with grounth truth for evaluation
    all_outputs = ground_truth.merge(
        all_outputs, how='inner', on=list(predictions_index.columns)
    )
    
    return all_outputs


def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == "cosine":
        lr_adjust = {epoch: args.learning_rate /2 * (1 + math.cos(epoch / args.train_epochs * math.pi))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(
        self, checkpoint_folder, patience=7, verbose=False, 
        delta=0, best_model_name='checkpoint.pth'
    ):
        self.checkpoint_folder = checkpoint_folder
        self.best_model_path = os.path.join(checkpoint_folder, best_model_name)
        
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
            
        if os.path.exists(self.checkpoint_folder):
            os.makedirs(self.checkpoint_folder, exist_ok=True)
            
        torch.save(model.state_dict(), self.best_model_path)
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


# class StandardScaler():
#     def __init__(self, mean, std):
#         self.mean = mean
#         self.std = std

#     def transform(self, data):
#         return (data - self.mean) / self.std

#     def inverse_transform(self, data):
#         return (data * self.std) + self.mean

def load_content(args):
    if 'ETT' in args.data:
        file = 'ETT'
    else:
        file = args.data
    with open('./dataset/prompt_bank/{0}.txt'.format(file), 'r') as f:
        content = f.read()
    return content