import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from transformers import BertTokenizer, BertModel
from einops import rearrange
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from peft import get_peft_config, get_peft_model, get_peft_model_state_dict, LoraConfig, TaskType


class AccustumGPT2Model(GPT2Model):
    def forward(self, input_ids=None, labels=None, **kwargs):
        kwargs.setdefault("output_hidden_states", True)
        outputs = super().forward(input_ids, **kwargs)
        return outputs.last_hidden_state, outputs.hidden_states # final feat, intermidiate feat






