import torch, argparse
from exp.exp_long_term_forecasting import *
import random, json
import numpy as np
from run import set_random_seed, initial_setup
from utils.arg_utils import get_basic_parser

def main(args):
    initial_setup(args)
    set_random_seed(args.seed)

    print(f'Args in experiment: {args}')
    if args.itrs == 1:
        exp = Exp_Long_Term_Forecast(args)
        
        if os.path.exists(exp.best_model_path):
            print(f'Checkpoint exists already. Skipping...')
        else:
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
            
            if os.path.exists(exp.best_model_path):
                print(f'Checkpoint exists already. Skipping...')

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
    parser = get_basic_parser('CALF')

    parser.add_argument(
        '--model_id', default='ori', choices=['ori', 'dropAttn_keepWE', 
        'randomInit', 'llm_to_trsf', 'llm_to_attn']
    )
    parser.add_argument('--model', type=str, default='CALF',choices=['CALF'])
    
    # distillation loss
    parser.add_argument('--task_loss', type=str, default='l1', help='task loss function')
    parser.add_argument('--distill_loss', type=str, default='l1', help='distillation loss function')
    parser.add_argument('--logits_loss', type=str, default='l1', help='logits loss function')
    
    # the rest here is CALF related arguments
    parser.add_argument('--tmax', type=int, default=20)

    # lora
    parser.add_argument('--r', type=int, default=8)
    parser.add_argument('--lora_alpha', type=int, default=32)
    parser.add_argument('--lora_dropout', type=float, default=0.1)

    # align
    parser.add_argument('--word_embedding_path', type=str, default="./utils/wte_pca_500.pt")

    # loss weight
    parser.add_argument('--task_w', type=float, default=1.0)
    parser.add_argument('--feature_w', type=float, default=0.01)
    parser.add_argument('--logits_w', type=float, default=1.0)
    
    # gpt
    parser.add_argument('--gpt_layers', type=int, default=6, help='number of hidden layers in gpt')
    
    # Save Result in this file 
    parser.add_argument('--log_fine_name', type=str, default='CALF_result.txt')

    # Add nosise to wordEmb or Posi
    parser.add_argument('--noise_scale',required=False , type=float, default=-100)
    parser.add_argument('--bootstrap_eval',required=False , type=int, default=0)
    
    return parser

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)


