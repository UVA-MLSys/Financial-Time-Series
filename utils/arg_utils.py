import argparse

def get_basic_parser(
    name='TimeSeries Model', 
    non_stationary=False, timemixer=False
):
    parser = argparse.ArgumentParser(description=name)

    # basic config
    parser.add_argument('--test', action='store_true', help='test the model')
    parser.add_argument('--seed', type=int, default=2024, help='random seed')
    parser.add_argument('--result_path', type=str, default='results', help='result output folder')
    parser.add_argument('--disable_progress', action='store_true', help='do not show progress bar')

    # data loader
    parser.add_argument('--root_path', type=str, default='./data', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='Exchange_Rate_Report.csv', help='data file')
    parser.add_argument('--features', type=str, default='M', choices=['M', 'S', 'MS'],
        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--n_features', type=int, required=True, help='Number of input features')
    # parser.add_argument('--group_id', type=str, default='GROUP_ID', help='group id for multi-time series')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument(
        '--freq', type=str, default='d', choices=['s', 't', 'h', 'd', 'b', 'w', 'm', 'a'],
        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly, a:yearly], you can also use more detailed freq like 15min or 3h'
    )
    parser.add_argument('--no_scale', action='store_true', help='do not scale the dataset')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=24, help='prediction sequence length')
    # parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
    # parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)

    # model define
    parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    # parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    # parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    # parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=64, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=4, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=128, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=3, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    
    # TimeMixer parameters
    if timemixer:
        parser.add_argument('--channel_independence', type=int, default=1,
            help='1: channel dependence 0: channel independence for FreTS model')
        parser.add_argument('--decomp_method', type=str, default='moving_avg',
                            help='method of series decompsition, only support moving_avg or dft_decomp')
        parser.add_argument('--use_norm', type=int, default=1, help='whether to use normalize; True 1 False 0')
        parser.add_argument('--down_sampling_layers', type=int, default=0, help='num of down sampling layers')
        parser.add_argument('--down_sampling_window', type=int, default=1, help='down sampling window size')
        parser.add_argument('--down_sampling_method', type=str, default=None,
                            help='down sampling method, only support avg, max, conv')
        parser.add_argument('--seg_len', type=int, default=48,
                            help='the length of segmen-wise iteration of SegRNN')
    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itrs', type=int, default=1, help='experiments times')
    parser.add_argument('--itr_no', type=int, default=None, help='experiments number among itrs. 1<= itr_no <= itrs .')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default=None, help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')

    # GPU
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    # de-stationary projector params
    if non_stationary:
        parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                            help='hidden layer dimensions of projector (List)')
        parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')
        
    parser.add_argument('--dry_run', action='store_true', help='run only one batch for test')
    parser.add_argument('--percent', type=int, default=100, help='Percent of recent examples to keep for training, 0<= percent <= 100')
    return parser