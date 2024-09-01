from utils.tools import calculate_metrics
from tqdm import tqdm

import numpy as np
import torch, argparse, warnings, random
import pandas as pd
warnings.filterwarnings('ignore')

SEASONALITY_MAP = {
   "minutely": 1440,
   "10_minutes": 144,
   "half_hourly": 48,
   "hourly": 24,
   "daily": 7,
   "weekly": 1,
   "monthly": 12,
   "quarterly": 4,
   "yearly": 1
}

def initial_setup(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    args.use_gpu = True if torch.cuda.is_available() else False
    
    if args.use_gpu:
        device_address = 'cuda:'+str(args.gpu_loc)
    else:
        device_address = 'cpu'
        
    # args.enc_in = args.dec_in = args.c_out = args.n_features
    args.task_name = 'long_term_forecast'
    args.freq = 'a' # yearly frequency
    return device_address

def stringify_setting(args):
    setting = f"OFA_{args.model_id}_sl_{args.seq_len}_pl_{args.pred_len}_features_{args.features}"
    return setting

def get_parser():

    parser = argparse.ArgumentParser(description='OFA_LLM')

    parser.add_argument(
        '--model_id', default='ori', choices=['ori', 'removeLLM', 
        'randomInit', 'llm_to_trsf', 'llm_to_attn']
    )
    parser.add_argument('--seed', type=int, default=2024, help='random seed')
    parser.add_argument('--result_path', type=str, default='results', help='result output folder')
    parser.add_argument('--test', action='store_true', help='test the model')

    parser.add_argument('--root_path', type=str, default='./data')
    parser.add_argument('--data_path', type=str, default='Merged.csv')
    parser.add_argument('--data', type=str, default='custom')
    parser.add_argument('--features', type=str, default='S', choices=['M', 'S', 'MS'],)
    parser.add_argument('--freq',default='a')
    parser.add_argument('--target', type=str, default='OFFER_BALANCE')
    parser.add_argument('--embed', type=str, default='timeF')
    parser.add_argument('--percent', type=int, default=10)
    parser.add_argument('--all', type=int, default=0)

    parser.add_argument('--seq_len', type=int, default=5)
    parser.add_argument('--pred_len', type=int, default=1)
    parser.add_argument('--label_len', type=int, default=3)

    parser.add_argument('--decay_fac', type=float, default=0.75)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--train_epochs', type=int, default=10)
    parser.add_argument('--lradj', type=str, default='type1')
    parser.add_argument('--patience', type=int, default=3)

    parser.add_argument('--gpt_layers', type=int, default=3)
    parser.add_argument('--is_gpt', type=int, default=1)
    parser.add_argument('--e_layers', type=int, default=3)
    parser.add_argument('--d_model', type=int, default=768)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--d_ff', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--enc_in', type=int, default=1)
    parser.add_argument('--c_out', type=int, default=1)
    parser.add_argument('--patch_size', type=int, default=2)
    parser.add_argument('--kernel_size', type=int, default=3)

    parser.add_argument('--loss_func', type=str, default='mse')
    parser.add_argument('--pretrain', type=int, default=1)
    parser.add_argument('--freeze', type=int, default=1)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--max_len', type=int, default=-1)
    parser.add_argument('--hid_dim', type=int, default=16)
    parser.add_argument('--tmax', type=int, default=20)

    parser.add_argument('--itr', type=int, default=1)
    parser.add_argument('--cos', type=int, default=0)
    parser.add_argument('--train_ratio', type=float, default=1.0 , required=False)
    parser.add_argument('--save_file_name', type=str, default=None)
    parser.add_argument('--gpu_loc', type=int, default=0)
    parser.add_argument('--n_scale', type=float, default=-1)
    parser.add_argument('--method', type=str, default='')

    parser.add_argument('--no_scale', action='store_true', help='do not scale the dataset')
    parser.add_argument('--disable_progress', action='store_true', help='do not show progress bar')
    parser.add_argument('--dry_run', action='store_true', help='run only one batch for test')
    parser.add_argument('--overwrite', action='store_true', help='overwrite the result folder')
    
    return parser

def select_optimizer(model ,args):
    param_dict = [
        {"params": [p for n, p in model.named_parameters() if p.requires_grad and '_proj' in n], "lr": 1e-4},
        {"params": [p for n, p in model.named_parameters() if p.requires_grad and '_proj' not in n], "lr": args.learning_rate}
    ]
    model_optim = torch.optim.Adam([param_dict[1]], lr=args.learning_rate)
    loss_optim = torch.optim.Adam([param_dict[0]], lr=args.learning_rate)
    return model_optim, loss_optim

class SMAPE(torch.nn.Module):
    def __init__(self):
        super(SMAPE, self).__init__()
    def forward(self, pred, true):
        return torch.mean(200 * torch.abs(pred - true) / (torch.abs(pred) + torch.abs(true) + 1e-8))
    
def vali(model, vali_loader, criterion, args, device):
    total_loss = []
    f_dim = -1 if args.features == 'MS' else 0
    
    model.in_layer.eval()
    model.out_layer.eval()

    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(vali_loader)):
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float()
            
            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)

            outputs = model(batch_x)
            
            # encoder - decoder
            outputs = outputs[:, -args.pred_len:, f_dim:]
            batch_y = batch_y[:, -args.pred_len:, f_dim:].to(device)
            
            pred = outputs.detach().cpu()
            true = batch_y.detach().cpu()

            loss = criterion(pred, true)

            total_loss.append(loss)
    total_loss = np.average(total_loss)
    
    model.in_layer.train()
    model.out_layer.train()
        
    return total_loss

def test(model, test_data, test_loader, args, device):
    preds = []
    trues = []
    model.eval()
    f_dim = -1 if args.features == 'MS' else 0
    
    with torch.no_grad():
        for batch_x, batch_y, _, _ in tqdm(test_loader, desc=f'Testing {test_data.flag}'):
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            
            outputs = model(batch_x[:, -args.seq_len:, :])
            
            outputs = outputs[:, -args.pred_len:, f_dim:]
            batch_y = batch_y[:, -args.pred_len:, f_dim:].to(device)

            pred = outputs.detach().cpu().numpy()
            true = batch_y.detach().cpu().numpy()
            
            preds.append(pred)
            trues.append(true)

    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
        
    # This assumes single target
    assert type(test_data.target) == str or len(test_data.target) == 1 
    preds = preds.reshape(preds.shape[0], -1)
    trues = trues.reshape(trues.shape[0], -1)
    
    print(f'test shape: preds{preds.shape}, trues {trues.shape}')
    
    result_df = test_data.index
    if type(test_data.target) == str: 
        target = test_data.target
    else: target = test_data.target[0]
    
    if args.pred_len == 1:
        trues_columns = [target]
        preds_columns = ['Predicted_' + target]
    else:
        trues_columns = [f'{target}_{i}' for i in range(args.pred_len)]
        preds_columns = [f'Predicted_{target}_{i}' for i in range(args.pred_len)]
        
    result_df[trues_columns] = trues
    result_df[preds_columns] = preds
    
    df = []
    progress_bar = tqdm(
        result_df.groupby(test_data.group_id),
        desc="Upscaling preds and trues", 
        disable=args.disable_progress
    )
    for group_id, group_df in progress_bar:
        group_df.loc[:, trues_columns] = test_data.upscale_target(
            group_id, group_df[trues_columns].values
        )
        group_df.loc[:, preds_columns] = test_data.upscale_target(
            group_id, group_df[preds_columns].values
        )
        df.append(group_df)
    result_df = pd.concat(df, axis=0) 
    
    print('Removing negative values ...')
    result_df[result_df<0] = 0

    # preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
    # trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
    # print('test shape:', preds.shape, trues.shape)
    preds = result_df[preds_columns].values
    trues = result_df[trues_columns].values
    
    mae, mse, r2 = calculate_metrics(preds, trues)
    # print('mae:{:.4f}, mse:{:.4f}, rmse:{:.4f}, smape:{:.4f}, mases:{:.4f}'.format(mae, mse, rmse, smape, mases))
    print(f'mse:{mse:.5g}, mae:{mae:.5g}, r2:{r2:.5g}')

    return mse, mae, r2, result_df
