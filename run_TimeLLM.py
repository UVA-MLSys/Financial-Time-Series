from utils.tools import stringify_setting
from exp.exp_long_term_forecasting import *
import numpy as np
import torch, os, time, warnings, json, argparse
warnings.filterwarnings('ignore')

from run import set_random_seed, initial_setup

def load_content(args):
    df = pd.read_csv(os.path.join(args.root_path, 'prompt_bank.csv'))
    data_name = args.data_path.split('.')[0] 
    content = df[df['data']==data_name]['prompt'].values[0]
    return content

def main(args):
    initial_setup(args)
    set_random_seed(args.seed)
    args.content = load_content(args)

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

    parser = argparse.ArgumentParser(description='TimeLLM')

    parser.add_argument(
        '--model_id', default='ori', choices=['ori', 'removeLLM', 
        'randomInit', 'llm_to_trsf', 'llm_to_attn']
    )
    parser.add_argument('--model', type=str, default='TimeLLM', choices=['TimeLLM'])
    parser.add_argument('--seed', type=int, default=2024, help='random seed')
    parser.add_argument('--result_path', type=str, default='results', help='result output folder')
    parser.add_argument('--test', action='store_true', help='test the model')

    parser.add_argument('--root_path', type=str, default='./data')
    parser.add_argument('--data_path', type=str, default='Exchange_Rate_Report.csv')
    parser.add_argument('--data', type=str, default='custom')
    parser.add_argument('--features', type=str, default='M', choices=['M', 'S', 'MS'],)
    parser.add_argument('--n_features', type=int, required=True, help='Number of input features')
    parser.add_argument(
        '--freq', type=str, default='d', choices=['s', 't', 'h', 'd', 'b', 'w', 'm'],
        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h'
    )
    parser.add_argument('--target', type=str, default='OFFER_BALANCE')
    parser.add_argument('--embed', type=str, default='timeF')
    parser.add_argument('--percent', type=int, default=10)

    parser.add_argument('--seq_len', type=int, default=96)
    parser.add_argument('--pred_len', type=int, default=48)
    parser.add_argument('--label_len', type=int, default=24)

    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--des', type=str, default=None, help='exp description')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--itrs', type=int, default=1, help='experiments times')
    parser.add_argument('--itr_no', type=int, default=None, help='experiments number among itrs. 1<= itr_no <= itrs .')
    
    parser.add_argument('--train_epochs', type=int, default=10)
    parser.add_argument('--lradj', type=str, default='type1')
    parser.add_argument('--patience', type=int, default=3)
    
    parser.add_argument('--e_layers', type=int, default=3)
    parser.add_argument('--d_model', type=int, default=768)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--d_ff', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--enc_in', type=int, default=1)
    parser.add_argument('--c_out', type=int, default=1)
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in encoder')
    
    parser.add_argument('--patch_len', type=int, default=16, help='patch length')
    parser.add_argument('--stride', type=int, default=8, help='stride')
    parser.add_argument('--prompt_domain', type=int, default=1, help='')
    parser.add_argument(
        '--llm_model', type=str, default='LLAMA', help='LLM model',
        choices=['LLAMA', 'GPT2', 'BERT']) # 
    parser.add_argument('--llm_dim', type=int, default='4096', help='LLM model dimension')# LLama7b:4096; GPT2-small:768; BERT-base:768
    parser.add_argument('--llm_layers', type=int, default=6)
    
    parser.add_argument('--tmax', type=int, default=20)
    parser.add_argument('--cos', type=int, default=0)
    
    # GPU
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    parser.add_argument('--no_scale', action='store_true', help='do not scale the dataset')
    parser.add_argument('--disable_progress', action='store_true', help='do not show progress bar')
    parser.add_argument('--dry_run', action='store_true', help='run only one batch for test')
    parser.add_argument('--overwrite', action='store_true', help='overwrite the result folder')
    
    return parser

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)