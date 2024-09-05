from utils.tools import stringify_setting
from exp.exp_long_term_forecasting import *
import numpy as np
import torch, os, time, warnings, json, argparse
warnings.filterwarnings('ignore')
from utils.arg_utils import get_basic_parser

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

    parser = get_basic_parser("TimeLLM")

    parser.add_argument(
        '--model_id', default='ori', choices=['ori', 'removeLLM', 
        'randomInit', 'llm_to_trsf', 'llm_to_attn']
    )
    parser.add_argument('--model', type=str, default='TimeLLM', choices=['TimeLLM'])
    
    parser.add_argument('--patch_len', type=int, default=16, help='patch length')
    parser.add_argument('--stride', type=int, default=8, help='stride')
    parser.add_argument('--prompt_domain', type=int, default=1, help='')
    parser.add_argument(
        '--llm_model', type=str, default='GPT2', help='LLM model',
        choices=['LLAMA', 'GPT2', 'BERT']) # 
    parser.add_argument('--llm_dim', type=int, default='768', 
        help='LLM model dimension. LLama7b:4096; GPT2-small:768; BERT-base:768')
    parser.add_argument('--llm_layers', type=int, default=6)
    
    return parser

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)