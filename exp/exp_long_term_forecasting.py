from data_provider.data_factory import *
from exp.exp_basic import *
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import calculate_metrics
import torch
from datetime import datetime
import torch.nn as nn
from torch import optim
import os, time, warnings, gc
import pandas as pd
import numpy as np
from tqdm import tqdm
from utils.tools import align_predictions
from utils.distillationLoss import DistillationLoss

warnings.filterwarnings('ignore')

class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)
        self.log(f"\nExperiment begins at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
    @property
    def log_file(self): return 'results.txt'
        
    def log(self, msg, verbose=True):
        if verbose: print(msg)
        with open(
                os.path.join(self.output_folder, self.log_file), 'a'
            ) as output_file:
            output_file.write(msg+'\n')

    def _select_optimizer(self):
        if self.args.model == 'CALF':
            param_dict = [
                {"params": [p for n, p in self.model.named_parameters() if p.requires_grad and '_proj' in n], "lr": 1e-4},
                {"params": [p for n, p in self.model.named_parameters() if p.requires_grad and '_proj' not in n], "lr": self.args.learning_rate}
            ]
            model_optim = optim.Adam([param_dict[1]], lr=self.args.learning_rate)
            loss_optim = optim.Adam([param_dict[0]], lr=self.args.learning_rate)
            return model_optim, loss_optim
        else:
            model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
            return model_optim

    def _select_criterion(self):
        if self.args.model == 'CALF':
            criterion = DistillationLoss(
                self.args.distill_loss, 
                self.args.logits_loss, 
                self.args.task_loss, 
                self.args.task_name, 
                self.args.feature_w, 
                self.args.logits_w, 
                self.args.task_w,
                self.args.features, 
                self.args.pred_len
            )
        elif self.args.model == 'OFA':
            criterion = nn.L1Loss()
        else: criterion = nn.MSELoss()
        
        return criterion
    
    def _select_lr_scheduler(self, optimizer):
        if self.args.model in ['CALF', 'OFA']:
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.args.tmax, 
                eta_min=1e-8, verbose=True
            )
        else:
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, patience=1, factor=0.1, 
                verbose=True, min_lr=5e-6
            )

    def vali(self, vali_loader, criterion):
        total_loss = []
        
        if self.args.model == 'CALF':
            self.model.in_layer.eval()
            self.model.out_layer.eval()
            self.model.time_proj.eval()
            self.model.text_proj.eval()
            
            criterion = nn.MSELoss()
        elif self.args.model == 'OFA':
            self.model.in_layer.eval()
            self.model.out_layer.eval()
        else:
            self.model.eval()
        f_dim = -1 if self.args.features == 'MS' else 0
        
        progress_bar =tqdm(
            vali_loader, desc=f'Validation', 
            disable=self.args.disable_progress
        )
        
        with torch.no_grad():
            for (batch_x, batch_y, batch_x_mark, batch_y_mark) in progress_bar:
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                
                if self.args.model == 'CALF':
                    outputs = self.model(batch_x)['outputs_time']
                elif self.args.model == 'OFA':
                    outputs = self.model(batch_x)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    if self.args.output_attention:
                        outputs = outputs[0]
                    
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)
                total_loss.append(loss)
        
        total_loss = np.average(total_loss)
        
        if self.args.model == 'CALF':
            self.model.in_layer.train()
            self.model.out_layer.train()
            self.model.time_proj.train()
            self.model.text_proj.train()
        elif self.args.model == 'OFA':
            self.model.in_layer.train()
            self.model.out_layer.train()
        else: 
            self.model.train()
            
        return total_loss

    def train(self):
        if self.args.percent == 0:
            print('Zero shot learning, no need to train')
            return
        
        _, train_loader = self.get_data(flag='train')
        _, vali_loader = self.get_data(flag='val')

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(
            self.output_folder, 
            patience=self.args.patience, verbose=True,
            best_model_name=self.best_model_name
        )

        if self.args.model == 'CALF':
            model_optim, loss_optim = self._select_optimizer()
        else: model_optim = self._select_optimizer()
            
        criterion = self._select_criterion()
        lr_scheduler = self._select_lr_scheduler(model_optim)
        
        f_dim = -1 if self.args.features == 'MS' else 0
        
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                if self.args.model == 'CALF': 
                    loss_optim.zero_grad()
                
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat(
                    [batch_y[:, :self.args.label_len, :], dec_inp], dim=1
                ).float().to(self.device)

                if self.args.model in ['CALF', 'OFA']:
                    outputs = self.model(batch_x)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    if self.args.output_attention:
                        outputs = outputs[0]
                        
                # only CALF model has dictionary output
                if self.args.model != 'CALF':
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())

                if (i + 1) % 1000 == 0:
                    print(f"\titers: {i + 1} | loss: {loss.item():.5g}")
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print(f'\tspeed: {speed:.4g}s/iter; left time: {left_time:.4g}s')
                    iter_count = 0
                    time_now = time.time()
                    
                loss.backward()
                model_optim.step()
                if self.args.model == 'CALF': loss_optim.step()
            
            train_loss = np.average(train_loss)
            
            val_loss = self.vali(vali_loader, criterion)

            print(f"Epoch: {epoch + 1} | Time: {time.time() - epoch_time:0.3g} s | Train Loss: {train_loss:.5g} Vali Loss: {val_loss:.5g}")
            early_stopping(val_loss, self.model)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            lr_scheduler.step(val_loss)
            gc.collect()
        
        time_per_epoch = (time.time() - time_now) / (epoch + 1)
        print(f"\nTraining completed at {datetime.today().strftime('%Y-%m-%d %H:%M:%S')}.")
        gc.collect()
            
        self.profile(time_per_epoch)
        self.load_best_model()
        return self.model
    
    def profile(self, time_per_epoch):
        # add if p.requires_grad to count only trainable parameters
        total_params = sum(p.numel() for p in self.model.parameters()) 
        self.log(f"Model parameters: {total_params}")
        
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**2
        # Get the current memory allocated by PyTorch on the GPU
        allocated_memory = torch.cuda.memory_allocated(0) / 1024**2
        # Get the maximum memory allocated by PyTorch on the GPU
        max_allocated_memory = torch.cuda.max_memory_allocated(0) / 1024**2

        print(f"Total memory: {total_memory:.1f} MB")
        print(f"Allocated memory: {allocated_memory:.1f} MB")
        print(f"Max allocated memory: {max_allocated_memory:.1f} MB")
        # print(torch.cuda.memory_summary())
        
        self.log(f"Time per epoch: {time_per_epoch:.1f} sec.")
        self.log(f"Memory usage: Available {total_memory:.1f} MB, Allocated {allocated_memory:.1f} MB, Max allocated {max_allocated_memory:.1f} MB\n")
    
    def test(
        self, load_model:bool=False, flag='test', 
        evaluate=True, dump_output=False, 
        remove_negative=True
    ):
        test_data, test_loader = self.get_data(flag)
        
        # percent 0 is for zero-shot learning, no need to load model
        if (load_model or self.args.test) and self.args.percent > 0:
            self.load_best_model()
        else:
            print('No need to load model')
            
        disable_progress = self.args.disable_progress

        preds = []
        trues = []
        f_dim = -1 if self.args.features == 'MS' else 0

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(
                enumerate(test_loader), desc="Running inference",
                total=len(test_loader), disable=disable_progress
            ):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                
                if self.args.model == 'CALF':
                    outputs = self.model(batch_x)
                    outputs = outputs['outputs_time']
                elif self.args.model == 'OFA':
                    outputs = self.model(batch_x)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    
                    if self.args.output_attention: outputs = outputs[0]

                outputs = outputs[:, -self.args.pred_len:, f_dim:].detach().cpu().numpy()
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].detach().cpu().numpy()

                preds.append(outputs)
                trues.append(batch_y)

        # this line handles different size of batch. E.g. last batch can be < batch_size.
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        data_name = self.args.data_path.split('.')[0]

        print('Preds and Trues shape:', preds.shape, trues.shape)
        if evaluate:
            # calculate evaluations
            mae, rmse, _, _ = calculate_metrics(preds, trues)
            mse = rmse ** 2
            
            # dump results in the global file
            with open("result_long_term_forecast.txt", 'a') as f:
                f.write(data_name + " " + self.setting + "  " + flag + " scaled\n")
                f.write(f'mae:{mae:.5g}, rmse:{rmse:.5g}, mse:{mse:.5g}\n\n')

            # dump results in the respective result folder
            self.log(f'{flag} scaled -- mse:{mse:.5g}, mae:{mae:.5g}')
        
        # inverse transform and remove negatives
        print("Upscaling data and removing negatives...")
    
        for i in range(preds.shape[0]):
            # date = test_data.index.loc[i, 'date']
            scaler = test_data.scaler[i]
            preds[i] = test_data.inverse_transform(scaler, preds[i])
            trues[i] = test_data.inverse_transform(scaler, trues[i])
    
        if remove_negative:
            print("Removing negatives...")
            preds[preds<0] = 0
        # print('Trues ', trues)
        # print('Preds ', preds)
        
        if evaluate:
            # calculate evaluations
            mae, rmse, rmsle, smape = calculate_metrics(preds, trues)
            mse = rmse ** 2
            
            # dump results in the global file
            with open("result_long_term_forecast.txt", 'a') as f:
                f.write(data_name + " " + self.setting + "  " + flag + "\n")
                f.write(f'mae:{mae:.5g}, rmse:{rmse:.5g}, mse:{mse:.5g}, rmsle {rmsle:0.5g} smape {smape:0.5g}\n\n')

            # dump results in the respective result folder
            self.log(f'{flag} -- mse:{mse:.5g}, mae:{mae:.5g}, rmsle: {rmsle:0.5g} smape {smape:0.5g}\n')
                
        if dump_output:
            # get ground truth
            target, time_col = test_data.target, test_data.time_col
            selected_columns = [time_col]
            
            if type(test_data) == MultiTimeSeries: 
                selected_columns.append(test_data.group_id)
            
            if type(target) == list: selected_columns += target
            else: selected_columns.append(target)
            
            filepath = os.path.join(self.args.root_path, self.args.data_path)
            ground_truth = pd.read_csv(filepath)[selected_columns]
            
            if ground_truth[test_data.time_col].dtype == 'object':
                ground_truth[test_data.time_col] = pd.to_datetime(ground_truth[test_data.time_col])
                
            # output prediction into a csv file
            merged = align_predictions(
                ground_truth, preds, test_data, 
                remove_negative=False, upscale=False, 
                disable_progress=self.args.disable_progress
            )
            merged.round(4).to_csv(
                os.path.join(self.output_folder, f'{flag}.csv'), 
                index=False
            )
            
        gc.collect()