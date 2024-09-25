import argparse, os, torch, random, json
from exp.exp_long_term_forecasting import *
import numpy as np
from utils.arg_utils import get_basic_parser

def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def initial_setup(args):
    args.use_gpu = True if torch.cuda.is_available() else False
    if args.use_gpu:
        print(torch.cuda.get_device_name(0))
    
    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]
        
    args.enc_in = args.dec_in = args.c_out = args.n_features
    args.task_name = 'long_term_forecast'


def main(args):
    initial_setup(args)
    set_random_seed(args.seed)

    print(f'Args in experiment: {args}')
    if args.itrs == 1:
        exp = Exp_Long_Term_Forecast(args)
        
        if not args.test:
            if os.path.exists(exp.best_model_path):
                print(f'Checkpoint exists already. Skipping...')
            else:
                print('>>>>>>> start training :>>>>>>>>>>')
                exp.train()
        else:
            print('\n>>>>>>> testing :  <<<<<<<<<<<<<<<<<<<')
            exp.test(flag='test', dump_output=True)
    else:
        parent_seed = args.seed
        np.random.seed(parent_seed)
        experiment_seeds = np.random.randint(1e3, size=args.itrs)
        experiment_seeds = [int(seed) for seed in experiment_seeds]
        args.experiment_seeds = experiment_seeds
        original_itr = args.itr_no
        
        for itr_no in range(1, args.itrs+1):
            if (original_itr is not None) and original_itr != itr_no: continue
            
            args.seed = experiment_seeds[itr_no-1]
            print(f'\n>>>> itr_no: {itr_no}, seed: {args.seed} <<<<<<')
            set_random_seed(args.seed)
            args.itr_no = itr_no
            
            exp = Exp_Long_Term_Forecast(args)
            
            if not args.test:
                if os.path.exists(exp.best_model_path):
                    print(f'Checkpoint exists already. Skipping...')
                else:
                    print('>>>>>>> start training :>>>>>>>>>>')
                    exp.train()
            else:
                print('\n>>>>>>> testing :  <<<<<<<<<<<<<<<<<<<')
                exp.test(flag='test')
           
        data_name = args.data_path.split('.')[0] 
        config_filepath = os.path.join(
            args.result_path, data_name, 
            stringify_setting(args), 'config.json'
        )
        args.seed = parent_seed
        with open(config_filepath, 'w') as output_file:
            json.dump(vars(args), output_file, indent=4)

if __name__ == '__main__':
    parser = get_basic_parser(
        non_stationary=True, timemixer=True
    )
    parser.add_argument('--model', type=str, default='DLinear', 
        choices=list(Exp_Basic.model_dict.keys()), help='model name')
    args = parser.parse_args()
    main(args)
