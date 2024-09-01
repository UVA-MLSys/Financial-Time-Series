from data_provider.data_factory import data_provider
from utils.tools import EarlyStopping, adjust_learning_rate
from models.OFA import Model as OFA

import numpy as np
import torch, os, time, warnings
import numpy as np
warnings.filterwarnings('ignore')
from utils.ofa_utils import *

parser = get_parser()
args = parser.parse_args()

if args.save_file_name is not None : 
    log_fine_name = args.save_file_name
else: log_fine_name = 'OFA_LLM.txt'

device_address = initial_setup(args)
device = torch.device(device_address)

best_model_name = 'llm.pth'
setting = stringify_setting(args)
mses, maes, r2s = [], [], []
model = OFA(args, device , log_fine_name = log_fine_name)

for ii in range(args.itr):
    path = os.path.join(args.result_path, setting)
    if not os.path.exists(path):
        os.makedirs(path)
        
    best_model_path = os.path.join(path, best_model_name)
    
    # if train mode on and the file does not exist or the overwrite flag is set
    if not args.test and (not os.path.exists(best_model_path) or args.overwrite):
        time_now = time.time()
        train_data, train_loader = data_provider(args, 'train')
        vali_data, vali_loader = data_provider(args, 'val')
        train_steps = len(train_loader)
        
        params = model.parameters()
        model_optim = torch.optim.Adam(params, lr=args.learning_rate)

        early_stopping = EarlyStopping(
            checkpoint_folder=path, patience=args.patience, verbose=True,
            best_model_name=best_model_name
        )
        
        if args.loss_func == 'smape': criterion = SMAPE()

        criterion = torch.nn.L1Loss() # nn.MSELoss()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            model_optim, T_max=args.tmax, 
            eta_min=1e-8
        )
        f_dim = -1 if args.features == 'MS' else 0
        for epoch in range(args.train_epochs):

            iter_count = 0
            train_loss = []
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(
                enumerate(train_loader), total=len(train_loader)
            ):    
                if iter_count ==0 and i == 0 and epoch == 0: 
                    print('\n', args.data_path, batch_x.shape  , batch_y.shape, '\n')
                    
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(device)

                batch_y = batch_y.float().to(device)
                batch_x_mark = batch_x_mark.float().to(device)
                batch_y_mark = batch_y_mark.float().to(device)
                # print(batch_x.shape, batch_y.shape , )
                # [Batch, Channel, Output length]
                outputs = model(batch_x)
                
                # print(outputs.shape , batch_y.shape)
                outputs = outputs[:, -args.pred_len:, :]
                batch_y = batch_y[:, -args.pred_len:, :].to(device)
                    
                assert outputs.shape == batch_y.shape
                
                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())

                if (i + 1) % 1000 == 0:
                    print(f"\titers: {i+1}, epoch: {epoch + 1} | loss: {loss.item():.4f}")
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((args.train_epochs - epoch) * train_steps - i)
                    print(f'\tspeed: {speed:.2f}s/iter; left time: {left_time:.1f}s')
                    iter_count = 0
                    time_now = time.time()
                loss.backward()
                model_optim.step()
                
            print(f"Epoch: {epoch + 1} cost time: {time.time() - epoch_time:.0f}")
            
            train_loss = np.average(train_loss)
            vali_loss = vali(model, vali_loader, criterion, args, device)
            print(f"Epoch: {epoch + 1}, Steps: {train_steps} | Train Loss: {train_loss:.4g} Vali Loss: {vali_loss:.4g}")
                
            if args.cos:
                scheduler.step()
                print("lr = {:.3g}".format(model_optim.param_groups[0]['lr']))
            else:
                adjust_learning_rate(model_optim, epoch + 1, args)
            early_stopping(vali_loss, model)
            if early_stopping.early_stop:
                print("Early stopping")
                break
          
    print("------------------------------------")    
    model.load_state_dict(torch.load(best_model_path))
    test_data, test_loader = data_provider(args, 'test')
    mse, mae, r2, result_df = test(model, test_data, test_loader, args, device)
    
    mses.append(round(mse,5))
    maes.append(round(mae,5))
    r2s.append(round(r2,5))
    result_df.round(2).to_csv(os.path.join(path, 'test.csv'), index=False)

if len(maes)==0 : exit()
maes = np.array(maes)
mses = np.array(mses)
print("mse_mean = {:.4f}, mse_std = {:.4f}".format(np.mean(mses), np.std(mses)))
print("mae_mean = {:.4f}, mae_std = {:.4f}".format(np.mean(maes), np.std(maes)))
    
with open(log_fine_name , 'a') as f : 
    f.write(f"{args.model_id} test\n")
    # f.write("mae{}\n".format(str(maes)))
    # f.write("mse{}\n".format(str(mses)))
    content = ''
    for name, arr in zip(['mae', 'mse', 'r2'], [maes, mses, r2s]):
        content += f"{name}:{np.mean(arr):.4f}, std:{np.std(arr):.4f}\n"
    
    f.write(content)
    f.write('\n')
            
