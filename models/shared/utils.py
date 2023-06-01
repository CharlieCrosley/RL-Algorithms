import torch
import torch.nn as nn

def get_model_obj(model_name):
    """ Returns the model object given the model name """
    match model_name:
        case 'vpg':
            from models import VPG
            return VPG.VPG
        case 'trpo':
            from models import TRPO
            return TRPO.TRPO
    raise ValueError(f'Unknown model name: {model_name}')

""" def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01) """

def init_weights(module):
    if isinstance(module, nn.Linear):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
def init_weights2(module):
    if isinstance(module, nn.Linear):
        torch.nn.init.kaiming_normal_(module.weight)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)

def create_layers(n_input, n_output, hidden_layers=None, hidden_sizes=None, hidden_activation='relu', final_activation=None, frame_stack=1, bias=True):
    assert hidden_layers != None and hidden_sizes != None
    if hidden_layers == None:
        hidden_layers = len(hidden_sizes-1)
    elif hidden_sizes == None:
        hidden_sizes = [64] * (hidden_layers+1)

    layers = nn.ModuleList()
    layers.append(nn.Linear(frame_stack * n_input, hidden_sizes[0], bias=bias))
    layers.append(get_ativation_fn(hidden_activation))

    for i in range(hidden_layers):
        layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1], bias=bias))
        layers.append(get_ativation_fn(hidden_activation))

    layers.append(nn.Linear(hidden_sizes[-1], n_output, bias=bias))
    if final_activation is not None:
        layers.append(get_ativation_fn(final_activation))
    return layers

def get_ativation_fn(activation):
    match activation:
        case 'tanh':
            return nn.Tanh()
        case 'relu':
            return nn.ReLU()
        case 'sigmoid':
            return nn.Sigmoid()
        case 'softmax':
            return nn.Softmax(dim=-1)
        case _:
            raise ValueError(f'Unknown activation: {activation}')

def apply_parameter_update(parameters, new_param_flat):
    offset = 0
    for param in parameters:
        numel = param.numel()
        param.data.copy_(new_param_flat[offset:offset + numel].view(param.shape))
        offset += numel

def get_flat_params_from(model):
    params = []
    for param in model.parameters():
        params.append(param.data.view(-1))

    flat_params = torch.cat(params)
    return flat_params


def validate_cg(cg):
    from numpy.linalg import inv
    import numpy as np
    # This code validates conjugate gradients
    A = np.random.rand(8, 8)
    A = np.matmul(np.transpose(A), A)
    def f_Ax(x):
        return torch.matmul(torch.FloatTensor(A), x.view((-1, 1))).view(-1)
    b = np.random.rand(8)
    w = np.matmul(np.matmul(inv(np.matmul(np.transpose(A), A)),
                            np.transpose(A)), b.reshape((-1, 1))).reshape(-1)
    print("W", w)
    print("CG", cg(f_Ax, torch.FloatTensor(b)).numpy())