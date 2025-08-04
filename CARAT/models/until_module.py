

import logging
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import math

logger = logging.getLogger(__name__)

def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

def swish(x):
    return x * torch.sigmoid(x)

ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

class PreTrainedModel(nn.Module):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """
    def __init__(self, config, *inputs, **kwargs):
        super(PreTrainedModel, self).__init__()
        self.config = config

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, LayerNorm):
            if 'beta' in dir(module) and 'gamma' in dir(module):
                module.beta.data.zero_()
                module.gamma.data.fill_(1.0)
            else:
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def resize_token_embeddings(self, new_num_tokens=None):
        raise NotImplementedError

    @classmethod
    def init_preweight(cls, model, state_dict, prefix=None, task_config=None):
        old_keys = []
        new_keys = []
        for key in state_dict.keys():
            new_key = None
            if 'gamma' in key:
                new_key = key.replace('gamma', 'weight')
            if 'beta' in key:
                new_key = key.replace('beta', 'bias')
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)

        if prefix is not None:
            old_keys = []
            new_keys = []
            for key in state_dict.keys():
                old_keys.append(key)
                new_keys.append(prefix + key)
            for old_key, new_key in zip(old_keys, new_keys):
                state_dict[new_key] = state_dict.pop(old_key)

        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=''):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_from_state_dict(
                state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')

        load(model, prefix='')

        if prefix is None and (task_config is None or task_config.local_rank == 0):
            logger.info("-" * 20)
            if len(missing_keys) > 0:
                logger.info("Weights of {} not initialized from pretrained model: {}"
                            .format(model.__class__.__name__, "\n   " + "\n   ".join(missing_keys)))
            if len(unexpected_keys) > 0:
                logger.info("Weights from pretrained model not used in {}: {}"
                            .format(model.__class__.__name__, "\n   " + "\n   ".join(unexpected_keys)))
            if len(error_msgs) > 0:
                logger.error("Weights from pretrained model cause errors in {}: {}"
                             .format(model.__class__.__name__, "\n   " + "\n   ".join(error_msgs)))

        return model

    @property
    def dtype(self):
        """
        :obj:`torch.dtype`: The dtype of the module (assuming that all the module parameters have the same dtype).
        """
        try:
            return next(self.parameters()).dtype
        except StopIteration:
            # For nn.DataParallel compatibility in PyTorch 1.5
            def find_tensor_attributes(module: nn.Module):
                tuples = [(k, v) for k, v in module.__dict__.items() if torch.is_tensor(v)]
                return tuples

            gen = self._named_members(get_members_fn=find_tensor_attributes)
            first_tuple = next(gen)
            return first_tuple[1].dtype

    @classmethod
    def from_pretrained(cls, config, state_dict=None,  *inputs, **kwargs):
        """
        Instantiate a PreTrainedModel from a pre-trained model file or a pytorch state dict.
        Download and cache the pre-trained model file if needed.
        """
        # Instantiate model.
        model = cls(config, *inputs, **kwargs)
        if state_dict is None:
            return model
        model = cls.init_preweight(model, state_dict)

        return model


def getBinaryTensor(imgTensor, boundary = 0.35):
    one = torch.ones_like(imgTensor).fill_(1)
    zero = torch.zeros_like(imgTensor).fill_(0)
    return torch.where(imgTensor > boundary, one, zero)


class CTCModule(nn.Module): #
    def __init__(self, in_dim, out_seq_len):
        '''
        This module is performing alignment from A (e.g., audio) to B (e.g., text).
        :param in_dim: Dimension for input modality A
        :param out_seq_len: Sequence length for output modality B
        '''
        super(CTCModule, self).__init__()
        # Use LSTM for predicting the position from A to B
        self.pred_output_position_inclu_blank = nn.LSTM(in_dim, out_seq_len+1, num_layers=2, batch_first=True) # 1 denoting blank
        
        self.out_seq_len = out_seq_len
        
        self.softmax = nn.Softmax(dim=2)
    def forward(self, x):
        '''
        :input x: Input with shape [batch_size x in_seq_len x in_dim]
        '''
        # NOTE that the index 0 refers to blank. 
        pred_output_position_inclu_blank, _ = self.pred_output_position_inclu_blank(x)
        # print(x.shape, pred_output_position_inclu_blank.shape)
        prob_pred_output_position_inclu_blank = self.softmax(pred_output_position_inclu_blank) # batch_size x in_seq_len x out_seq_len+1
        prob_pred_output_position = prob_pred_output_position_inclu_blank[:, :, 1:] # batch_size x in_seq_len x out_seq_len
        prob_pred_output_position = prob_pred_output_position.transpose(1,2) # batch_size x out_seq_len x in_seq_len
        # print(prob_pred_output_position.shape)
        pseudo_aligned_out = torch.bmm(prob_pred_output_position, x) # batch_size x out_seq_len x in_dim
        
        # pseudo_aligned_out is regarded as the aligned A (w.r.t B)
        return pseudo_aligned_out, (pred_output_position_inclu_blank)
    

class AdaptiveCTCModule(nn.Module):
    def __init__(self, in_dim, min_target_len=50, max_target_len=400, max_position_embeddings=512):
        super(AdaptiveCTCModule, self).__init__()
        self.in_dim = in_dim
        self.min_target_len = min_target_len
        self.max_target_len = max_target_len
        self.max_position_embeddings = max_position_embeddings  # Store the position embedding limit
        
        # Adaptive projection layers maintaining original dimension
        self.hidden_dim = min(512, in_dim * 2)
        self.fc1 = nn.Linear(in_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.dropout = nn.Dropout(0.1)
        self.activation = nn.ReLU()
        
        # TfEmbeddings receives the expected original dimension
        self.final_projection = nn.Linear(self.hidden_dim, in_dim)

    def forward(self, x, target_length=None):
        # x shape: [batch_size, in_seq_len, in_dim]
        batch_size, in_seq_len, in_dim = x.shape
        
        # Calculate adaptive target length based on batch statistics
        if target_length is None:
            # Use 90th percentile of current batch lengths + small buffer
            batch_max = in_seq_len
            target_length = min(
                max(int(batch_max * 1.1), self.min_target_len),  # 10% buffer
                self.max_target_len
            )
        
        # Pass through feature transformation layers
        h1 = self.activation(self.fc1(x))  # [batch_size, in_seq_len, hidden_dim]
        h1 = self.dropout(h1)
        
        h2 = self.activation(self.fc2(h1))  # [batch_size, in_seq_len, hidden_dim]
        h2 = self.dropout(h2)
        
        # Project to standard output dimension
        features = self.final_projection(h2)  # [batch_size, in_seq_len, in_dim]
        
        # Adaptive temporal alignment using interpolation
        if in_seq_len != target_length:
            # Transpose for interpolation: [batch_size, in_dim, in_seq_len]
            features_t = features.transpose(1, 2)
            
            # Interpolate to target length
            aligned_features = F.interpolate(
                features_t, 
                size=target_length, 
                mode='linear', 
                align_corners=False
            )
            
            # Transpose back: [batch_size, target_length, in_dim]
            output = aligned_features.transpose(1, 2)
        else:
            output = features
        
        # Generate position encodings for aligned sequence
        # Position ids should not exceed max_position_embeddings
        effective_target_length = min(target_length, self.max_position_embeddings)
        position_ids = torch.arange(effective_target_length, dtype=torch.long, device=x.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        # If target_length > max_position_embeddings, truncate the output to match position_ids
        if target_length > self.max_position_embeddings:
            print(f"WARNING: target_length ({target_length}) > max_position_embeddings ({self.max_position_embeddings})")
            print(f"   Truncating output to {self.max_position_embeddings} to prevent CUDA indexing error")
            output = output[:, :self.max_position_embeddings, :]
        
        return output, position_ids
    

class MLAttention(nn.Module):
    def __init__(self, label_num, hidden_size):
        super(MLAttention, self).__init__()
        self.attention = nn.Linear(hidden_size, label_num, bias=False)
        nn.init.xavier_uniform_(self.attention.weight)

    def forward(self, inputs, masks):
        masks = torch.unsqueeze(masks, 1)
        attention = self.attention(inputs).transpose(1,2).masked_fill(~masks, -np.inf)
        attention = F.softmax(attention, -1)
        return attention @ inputs, attention


class MLLinear(nn.Module):
    def __init__(self, state_list, output_size):
        super(MLLinear, self).__init__()
        # print('hello', state_list)
        self.linear = nn.ModuleList(nn.Linear(in_s, out_s)
                                    for in_s, out_s in zip(state_list[:-1], state_list[1:]))
        for linear in self.linear:
            nn.init.xavier_uniform_(linear.weight)
        self.output = nn.Linear(state_list[-1], output_size)
        nn.init.xavier_uniform_(self.output.weight)

    def forward(self, inputs):
        linear_out = inputs
        for linear in self.linear:
            linear_out = F.relu(linear(linear_out))
        return torch.squeeze(self.output(linear_out), -1)