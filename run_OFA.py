from exp.exp_long_term_forecasting import *
import warnings
warnings.filterwarnings('ignore')

from run import main, get_basic_parser
            
def get_parser():
    parser = get_basic_parser("OFA LLM")
    
    parser.add_argument(
        '--model_id', default='ori', choices=['ori']
    )
    parser.add_argument('--model', type=str, default='OFA', choices=['OFA'])

    parser.add_argument('--gpt_layers', type=int, default=6)
    parser.add_argument('--is_gpt', type=int, default=1)
    parser.add_argument('--patch_size', type=int, default=16)

    parser.add_argument('--pretrain', type=int, default=1)
    parser.add_argument('--freeze', type=int, default=1)
    parser.add_argument('--stride', type=int, default=8)
    parser.add_argument('--max_len', type=int, default=-1)
    parser.add_argument('--hid_dim', type=int, default=16)
    parser.add_argument('--tmax', type=int, default=10)
    parser.add_argument('--n_scale', type=float, default=-1)
    
    return parser

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)