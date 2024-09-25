from exp.exp_long_term_forecasting import *
from run import main, get_basic_parser
    
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


