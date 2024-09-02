import os
import torch
from models import Transformer, DLinear, MICN, Crossformer, OFA, CALF, SegRNN, iTransformer, TimeMixer, PatchTST, TimesNet
from data_provider.data_factory import *
from utils.tools import stringify_setting

class Exp_Basic(object):
    model_dict = {
        'Transformer': Transformer,
        'DLinear': DLinear,
        'MICN': MICN,
        'Crossformer': Crossformer,
        'iTransformer': iTransformer,
        'TimeMixer': TimeMixer,
        'SegRNN': SegRNN,
        'PatchTST': PatchTST,
        'TimesNet': TimesNet,
        'CALF': CALF,
        'OFA': OFA,
    }
    
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)
        
        self.setting = stringify_setting(args)
        
        # set output folder
        data_name = args.data_path.split('.')[0]
        if args.itr_no is not None:
            self.output_folder = os.path.join(
                args.result_path, data_name, 
                self.setting, str(args.itr_no)
            )
        else:
            self.output_folder = os.path.join(
                args.result_path, data_name, self.setting
            )
            
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder, exist_ok=True)
        print(f'Experiments will be saved in {self.output_folder}')
        
        self.dataset_map = {}

    def _build_model(self):
        Model = self.model_dict[self.args.model].Model
        if self.args.model in ['CALF', 'OFA']:
            model = Model(self.args, self.device).float()
        else: model = Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = torch.nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device(f'cuda:{self.args.gpu}')
            print(f'Use GPU: cuda:{self.args.gpu}')
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def get_data(self, flag) -> tuple[Dataset_Custom, DataLoader]:
        if flag not in self.dataset_map:
            self.dataset_map[flag] = data_provider(self.args, flag)
        
        return self.dataset_map[flag]

    def val(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
    
    @property
    def best_model_name(self):
        if self.args.model in ['CALF', 'OFA', 'TimeLLM']:
            return 'llm.pth'
        else:
            return 'checkpoint.pth'
        
    @property
    def best_model_path(self):
        return os.path.join(self.output_folder, self.best_model_name)
        
    
    def load_best_model(self):
        print(f'Loading model from {self.best_model_path}')
        if self.model:
            self.model.load_state_dict(torch.load(self.best_model_path))
        else:
            self.model = self._build_model()
            
        # load to device memory
        self.model.to(self.device)