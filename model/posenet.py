import torch
import torch.nn as nn
from model.common import make_c2w, convert3x4_4x4
import numpy as np
import os,sys,time
import torch.nn.functional as torch_F
import torchvision


def get_layer_dims(layers):
    # return a list of tuples (k_in,k_out)
    return list(zip(layers[:-1],layers[1:]))
    
class TransNet(torch.nn.Module):

    def __init__(self,cfg):
        super().__init__()
        self.cfg = cfg
        self.input_t_dim = 11 
        self.define_network(cfg)
        self.max_index = cfg['pose']['max_index'] # consider training and testing
        self.min_index = 0
        

    def define_network(self,cfg):
        self.mlp_transnet = torch.nn.ModuleList()
        L = get_layer_dims(cfg['pose']['layers_feat'])

        for li,(k_in,k_out) in enumerate(L):
            if li==0: k_in = self.input_t_dim
            if li in self.cfg['pose']['skip']: k_in += self.input_t_dim
            if li==len(L)-1: k_out = 3
            linear = torch.nn.Linear(k_in,k_out)
            self.tensorflow_init_weights(linear,out="small" if li==len(L)-1 else "all")
            self.mlp_transnet.append(linear)

    def tensorflow_init_weights(self,linear,out=None):
        # use Xavier init instead of Kaiming init
        relu_gain = torch.nn.init.calculate_gain("relu") # sqrt(2)
        if out=="all":
            torch.nn.init.xavier_uniform_(linear.weight)
        elif out=="first":
            torch.nn.init.xavier_uniform_(linear.weight[:1])
            torch.nn.init.xavier_uniform_(linear.weight[1:],gain=relu_gain)
        elif out=="small":
            torch.nn.init.uniform_(linear.weight, b = 1e-6)
        else:
            torch.nn.init.xavier_uniform_(linear.weight,gain=relu_gain)
        torch.nn.init.zeros_(linear.bias)

    def forward(self, index):
        index = torch.tensor(index)
        index = index.reshape(-1,1).to(torch.float32).to(self.cfg['pose']['device'])
        index = 2*(index - self.min_index)/(self.max_index - self.min_index) - 1

        points_enc = self.positional_encoding(index, L= 5)
        points_enc = torch.cat([index,points_enc],dim=-1) # [B,...,6L+3]
        
        feat = points_enc
        # softplus_act = torch.nn.Softplus()
        # print("parameters of translation", list(self.mlp_translation.parameters())[0])
        for li,layer in enumerate(self.mlp_transnet):
            if li in self.cfg['pose']['skip']: feat = torch.cat([feat,points_enc],dim=-1)
            feat = layer(feat)
            if li==len(self.mlp_transnet)-1:
                feat = feat #torch.tanh(feat) * self.cfg['pose']['trans_scale']
            else:
                feat = torch_F.relu(feat) # softplus_act(translation_feat)

        return feat

    def positional_encoding(self,input,L): # [B,...,N]
        shape = input.shape
        freq = 2**torch.arange(L,dtype=torch.float32,device=self.cfg['pose']['device'])*np.pi # [L]
        spectrum = input[...,None]*freq # [B,...,N,L]
        sin,cos = spectrum.sin(),spectrum.cos() # [B,...,N,L]
        input_enc = torch.stack([sin,cos],dim=-2) # [B,...,N,2,L]
        input_enc = input_enc.view(*shape[:-1],-1) # [B,...,2NL]
        return input_enc


class RotsNet_(torch.nn.Module):

    def __init__(self,cfg):
        super().__init__()
        self.cfg = cfg
        self.input_t_dim = 11
        self.define_network(cfg)
        self.max_index = cfg['pose']['max_index'] # consider training and testing
        self.min_index = 0    

    def define_network(self,cfg):
        self.mlp_quad = torch.nn.ModuleList()
        L = get_layer_dims(cfg['pose']['layers_feat'])

        for li,(k_in,k_out) in enumerate(L):
            if li==0: k_in = self.input_t_dim
            if li in self.cfg['pose']['skip']: k_in += self.input_t_dim
            if li==len(L)-1: k_out = 3
            linear = torch.nn.Linear(k_in,k_out)
            self.tensorflow_init_weights(linear,out="small" if li==len(L)-1 else "all")
            self.mlp_quad.append(linear)

    def tensorflow_init_weights(self,linear,out=None):
        # use Xavier init instead of Kaiming init
        relu_gain = torch.nn.init.calculate_gain("relu") # sqrt(2)
        if out=="all":
            torch.nn.init.xavier_uniform_(linear.weight)
        elif out=="first":
            torch.nn.init.xavier_uniform_(linear.weight[:1])
            torch.nn.init.xavier_uniform_(linear.weight[1:],gain=relu_gain)
        elif out=="small":
            torch.nn.init.uniform_(linear.weight, b = 1e-6)
        else:
            torch.nn.init.xavier_uniform_(linear.weight,gain=relu_gain)
        torch.nn.init.zeros_(linear.bias)

    def forward(self, index):
        index = torch.tensor(index)
        index = index.reshape(-1,1).to(torch.float32).to(self.cfg['pose']['device'])
        index = 2*(index - self.min_index)/(self.max_index - self.min_index) - 1

        points_enc = self.positional_encoding(index, L= 5)
        points_enc = torch.cat([index,points_enc],dim=-1) # [B,...,6L+3]
        
        feat = points_enc
        # softplus_act = torch.nn.Softplus()
        # print("parameters of translation", list(self.mlp_translation.parameters())[0])
        for li,layer in enumerate(self.mlp_quad):
            if li in self.cfg['pose']['skip']: feat = torch.cat([feat,points_enc],dim=-1)
            feat = layer(feat)
            if li==len(self.mlp_quad)-1:
                feat = feat #torch_F.softplus(feat)
            else:
                feat = torch_F.relu(feat) # softplus_act(translation_feat)

        # norm_rots = torch.norm(feat,dim=-1)
        # rotation_feat_norm = feat / norm_rots[...,None]
        # rot_matrix = self.q_to_R(rotation_feat_norm)

        return feat

    def positional_encoding(self,input,L): # [B,...,N]
        shape = input.shape
        freq = 2**torch.arange(L,dtype=torch.float32,device=self.cfg['pose']['device'])*np.pi # [L]
        spectrum = input[...,None]*freq # [B,...,N,L]
        sin,cos = spectrum.sin(),spectrum.cos() # [B,...,N,L]
        input_enc = torch.stack([sin,cos],dim=-2) # [B,...,N,2,L]
        input_enc = input_enc.view(*shape[:-1],-1) # [B,...,2NL]
        return input_enc
   
    
class RotsNet_quad4(torch.nn.Module): # quad 4 

    def __init__(self,cfg):
        super().__init__()
        self.input_t_dim = 11
        self.cfg = cfg
        self.define_network(cfg)
        self.max_index = cfg['pose']['max_index'] # consider training and testing
        self.min_index = 0

    def q_to_R(self,q):
        # https://en.wikipedia.org/wiki/Rotation_matrix#Quaternion
        qa,qb,qc,qd = q.unbind(dim=-1)
        R = torch.stack([torch.stack([1-2*(qc**2+qd**2),2*(qb*qc-qa*qd),2*(qa*qc+qb*qd)],dim=-1),
                         torch.stack([2*(qb*qc+qa*qd),1-2*(qb**2+qd**2),2*(qc*qd-qa*qb)],dim=-1),
                         torch.stack([2*(qb*qd-qa*qc),2*(qa*qb+qc*qd),1-2*(qb**2+qc**2)],dim=-1)],dim=-2)
        return R

    def define_network(self,cfg):
        self.mlp_quad = torch.nn.ModuleList()
        L = get_layer_dims(cfg['pose']['layers_feat'])

        for li,(k_in,k_out) in enumerate(L):
            if li==0: k_in = self.input_t_dim
            if li in self.cfg['pose']['skip']: k_in += self.input_t_dim
            if li==len(L)-1: k_out = 4
            linear = torch.nn.Linear(k_in,k_out)
            self.tensorflow_init_weights(linear,out="small" if li==len(L)-1 else "all")
            self.mlp_quad.append(linear)

    def tensorflow_init_weights(self,linear,out=None):
        # use Xavier init instead of Kaiming init
        relu_gain = torch.nn.init.calculate_gain("relu") # sqrt(2)
        if out=="all":
            torch.nn.init.xavier_uniform_(linear.weight)
        elif out=="first":
            torch.nn.init.xavier_uniform_(linear.weight[:1])
            torch.nn.init.xavier_uniform_(linear.weight[1:],gain=relu_gain)
        elif out=="small":
            torch.nn.init.uniform_(linear.weight, b = 1e-6)
        else:
            torch.nn.init.xavier_uniform_(linear.weight,gain=relu_gain)
        torch.nn.init.zeros_(linear.bias)

    def forward(self, index):
        index = torch.tensor(index)
        index = index.reshape(-1,1).to(torch.float32).to(self.cfg['pose']['device'])
        index = 2*(index - self.min_index)/(self.max_index - self.min_index) - 1

        points_enc = self.positional_encoding(index, L= 5)
        points_enc = torch.cat([index,points_enc],dim=-1) # [B,...,6L+3]
        
        feat = points_enc
        # softplus_act = torch.nn.Softplus()
        # print("parameters of translation", list(self.mlp_translation.parameters())[0])
        for li,layer in enumerate(self.mlp_quad):
            if li in self.cfg['pose']['skip']: feat = torch.cat([feat,points_enc],dim=-1)
            feat = layer(feat)
            if li==len(self.mlp_quad)-1:
                feat = torch_F.tanh(feat)
            else:
                feat = torch_F.relu(feat) # softplus_act(translation_feat)

        norm_rots = torch.norm(feat,dim=-1)
        rotation_feat_norm = feat / norm_rots[...,None]
        # rot_matrix = self.q_to_R(rotation_feat_norm)

        return rotation_feat_norm

    def positional_encoding(self,input,L): # [B,...,N]
        shape = input.shape
        freq = 2**torch.arange(L,dtype=torch.float32,device=self.cfg['pose']['device'])*np.pi # [L]
        spectrum = input[...,None]*freq # [B,...,N,L]
        sin,cos = spectrum.sin(),spectrum.cos() # [B,...,N,L]
        input_enc = torch.stack([sin,cos],dim=-2) # [B,...,N,2,L]
        input_enc = input_enc.view(*shape[:-1],-1) # [B,...,2NL]
        return input_enc
   

 
class RotsNet_quad3(torch.nn.Module): # quad 4 

    def __init__(self,cfg):
        super().__init__()
        self.input_t_dim = 11
        self.cfg = cfg
        self.define_network(cfg)
        self.max_index = cfg['pose']['max_index'] # consider training and testing
        self.min_index = 0

    def q_to_R(self,q):
        # https://en.wikipedia.org/wiki/Rotation_matrix#Quaternion
        qa,qb,qc,qd = q.unbind(dim=-1)
        R = torch.stack([torch.stack([1-2*(qc**2+qd**2),2*(qb*qc-qa*qd),2*(qa*qc+qb*qd)],dim=-1),
                         torch.stack([2*(qb*qc+qa*qd),1-2*(qb**2+qd**2),2*(qc*qd-qa*qb)],dim=-1),
                         torch.stack([2*(qb*qd-qa*qc),2*(qa*qb+qc*qd),1-2*(qb**2+qc**2)],dim=-1)],dim=-2)
        return R

    def define_network(self,cfg):
        self.mlp_quad = torch.nn.ModuleList()
        L = get_layer_dims(cfg['pose']['layers_feat'])

        for li,(k_in,k_out) in enumerate(L):
            if li==0: k_in = self.input_t_dim
            if li in self.cfg['pose']['skip']: k_in += self.input_t_dim
            if li==len(L)-1: k_out = 3
            linear = torch.nn.Linear(k_in,k_out)
            self.tensorflow_init_weights(linear,out="small" if li==len(L)-1 else "all")
            self.mlp_quad.append(linear)

    def tensorflow_init_weights(self,linear,out=None):
        # use Xavier init instead of Kaiming init
        relu_gain = torch.nn.init.calculate_gain("relu") # sqrt(2)
        if out=="all":
            torch.nn.init.xavier_uniform_(linear.weight)
        elif out=="first":
            torch.nn.init.xavier_uniform_(linear.weight[:1])
            torch.nn.init.xavier_uniform_(linear.weight[1:],gain=relu_gain)
        elif out=="small":
            torch.nn.init.uniform_(linear.weight, b = 1e-6)
        else:
            torch.nn.init.xavier_uniform_(linear.weight,gain=relu_gain)
        torch.nn.init.zeros_(linear.bias)

    def forward(self, index):
        index = torch.tensor(index)
        index = index.reshape(-1,1).to(torch.float32).to(self.cfg['pose']['device'])
        index = 2*(index - self.min_index)/(self.max_index - self.min_index) - 1

        points_enc = self.positional_encoding(index, L= 5)
        points_enc = torch.cat([index,points_enc],dim=-1) # [B,...,6L+3]
        
        feat = points_enc
        # softplus_act = torch.nn.Softplus()
        # print("parameters of translation", list(self.mlp_translation.parameters())[0])
        for li,layer in enumerate(self.mlp_quad):
            if li in self.cfg['pose']['skip']: feat = torch.cat([feat,points_enc],dim=-1)
            feat = layer(feat)
            if li==len(self.mlp_quad)-1:
                feat = feat # torch_F.tanh(feat)
            else:
                feat = torch_F.relu(feat) # softplus_act(translation_feat)

        # norm_rots = torch.norm(feat,dim=-1)
        # rotation_feat_norm = feat / norm_rots[...,None]
        # rot_matrix = self.q_to_R(rotation_feat_norm)

        return feat

    def positional_encoding(self,input,L): # [B,...,N]
        shape = input.shape
        freq = 2**torch.arange(L,dtype=torch.float32,device=self.cfg['pose']['device'])*np.pi # [L]
        spectrum = input[...,None]*freq # [B,...,N,L]
        sin,cos = spectrum.sin(),spectrum.cos() # [B,...,N,L]
        input_enc = torch.stack([sin,cos],dim=-2) # [B,...,N,2,L]
        input_enc = input_enc.view(*shape[:-1],-1) # [B,...,2NL]
        return input_enc
    


class RotsNet_so3(torch.nn.Module): # quad 4 

    def __init__(self,cfg):
        super().__init__()
        self.input_t_dim = 11
        self.cfg = cfg
        self.define_network(cfg)
        self.max_index = cfg['pose']['max_index'] # consider training and testing
        self.min_index = 0

    def q_to_R(self,q):
        # https://en.wikipedia.org/wiki/Rotation_matrix#Quaternion
        qa,qb,qc,qd = q.unbind(dim=-1)
        R = torch.stack([torch.stack([1-2*(qc**2+qd**2),2*(qb*qc-qa*qd),2*(qa*qc+qb*qd)],dim=-1),
                         torch.stack([2*(qb*qc+qa*qd),1-2*(qb**2+qd**2),2*(qc*qd-qa*qb)],dim=-1),
                         torch.stack([2*(qb*qd-qa*qc),2*(qa*qb+qc*qd),1-2*(qb**2+qc**2)],dim=-1)],dim=-2)
        return R

    def define_network(self,cfg):
        self.mlp_quad = torch.nn.ModuleList()
        L = get_layer_dims(cfg['pose']['layers_feat'])

        for li,(k_in,k_out) in enumerate(L):
            if li==0: k_in = self.input_t_dim
            if li in self.cfg['pose']['skip']: k_in += self.input_t_dim
            if li==len(L)-1: k_out = 3
            linear = torch.nn.Linear(k_in,k_out)
            self.tensorflow_init_weights(linear,out="small" if li==len(L)-1 else "all")
            self.mlp_quad.append(linear)

    def tensorflow_init_weights(self,linear,out=None):
        # use Xavier init instead of Kaiming init
        relu_gain = torch.nn.init.calculate_gain("relu") # sqrt(2)
        if out=="all":
            torch.nn.init.xavier_uniform_(linear.weight)
        elif out=="first":
            torch.nn.init.xavier_uniform_(linear.weight[:1])
            torch.nn.init.xavier_uniform_(linear.weight[1:],gain=relu_gain)
        elif out=="small":
            torch.nn.init.uniform_(linear.weight, b = 1e-6)
        else:
            torch.nn.init.xavier_uniform_(linear.weight,gain=relu_gain)
        torch.nn.init.zeros_(linear.bias)

    def forward(self, index):
        index = torch.tensor(index)
        index = index.reshape(-1,1).to(torch.float32).to(self.cfg['pose']['device'])
        index = 2*(index - self.min_index)/(self.max_index - self.min_index) - 1

        points_enc = self.positional_encoding(index, L= 5)
        points_enc = torch.cat([index,points_enc],dim=-1) # [B,...,6L+3]
        
        feat = points_enc
        # softplus_act = torch.nn.Softplus()
        # print("parameters of translation", list(self.mlp_translation.parameters())[0])
        for li,layer in enumerate(self.mlp_quad):
            if li in self.cfg['pose']['skip']: feat = torch.cat([feat,points_enc],dim=-1)
            feat = layer(feat)
            if li==len(self.mlp_quad)-1:
                feat = torch.nn.functional.softplus(feat)
            else:
                feat = torch_F.relu(feat) # softplus_act(translation_feat)

        # norm_rots = torch.norm(feat,dim=-1)
        # rotation_feat_norm = feat / norm_rots[...,None]
        # rot_matrix = self.q_to_R(rotation_feat_norm)

        return feat

    def positional_encoding(self,input,L): # [B,...,N]
        shape = input.shape
        freq = 2**torch.arange(L,dtype=torch.float32,device=self.cfg['pose']['device'])*np.pi # [L]
        spectrum = input[...,None]*freq # [B,...,N,L]
        sin,cos = spectrum.sin(),spectrum.cos() # [B,...,N,L]
        input_enc = torch.stack([sin,cos],dim=-2) # [B,...,N,2,L]
        input_enc = input_enc.view(*shape[:-1],-1) # [B,...,2NL]
        return input_enc
    
