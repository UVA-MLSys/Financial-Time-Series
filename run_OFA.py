from data_provider.data_factory import data_provider
from utils.tools import EarlyStopping, adjust_learning_rate, stringify_setting
from exp.exp_long_term_forecasting import *
import numpy as np
from utils.arg_utils import get_basic_parser
import torch, os, time, warnings, json, argparse
warnings.filterwarnings('ignore')

from run import set_random_seed, initial_setup

def main(args):
    initial_setup(args)
    set_random_seed(args.seed)

    print(f'Args in experiment: {args}')
    if args.itrs == 1:
        exp = Exp_Long_Term_Forecast(args)
        if not args.test:
            print('>>>>>>> start training :>>>>>>>>>>')
            exp.train()

        print('\n>>>>>>> testing :  <<<<<<<<<<<<<<<<<<<')
        exp.test(flag='test')
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
                print('>>>>>>> start training :>>>>>>>>>>')
                exp.train()
            
            # print('\n>>>>>>> Evaluate train data :  <<<<<<<<<<<<<<<')
            # exp.test(load_model=True, flag='train')

            # print('\n>>>>>>> validating :  <<<<<<<<<<<<<<<')
            # exp.test(flag='val')

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
            
def get_parser():
    parser = get_basic_parser("OFA LLM")
    
    parser.add_argument(
        '--model_id', default='ori', choices=['ori', 'removeLLM', 
        'randomInit', 'llm_to_trsf', 'llm_to_attn']
    )
    parser.add_argument('--model', type=str, default='OFA', choices=['OFA'])

    parser.add_argument('--gpt_layers', type=int, default=6)
    parser.add_argument('--is_gpt', type=int, default=1)
    parser.add_argument('--d_model', type=int, default=768)
    parser.add_argument('--dropout', type=float, default=0.1)

    parser.add_argument('--patch_size', type=int, default=16)
    parser.add_argument('--kernel_size', type=int, default=25)

    parser.add_argument('--pretrain', type=int, default=1)
    parser.add_argument('--freeze', type=int, default=1)
    parser.add_argument('--stride', type=int, default=8)
    parser.add_argument('--max_len', type=int, default=-1)
    parser.add_argument('--hid_dim', type=int, default=16)
    parser.add_argument('--tmax', type=int, default=20)

    parser.add_argument('--cos', type=int, default=1)
    parser.add_argument('--n_scale', type=float, default=-1)
    
    return parser

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)