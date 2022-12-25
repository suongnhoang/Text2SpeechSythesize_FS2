import torch
import torch.nn as nn
import numpy as np

import transformer.Constants as Constants
from .Layers import FFTBlock
from text.symbols import symbols


def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    """ Sinusoid position encoding table """

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array(
        [get_posi_angle_vec(pos_i) for pos_i in range(n_position)]
    )

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.0

    return torch.FloatTensor(sinusoid_table)


class Encoder(nn.Module):
    """ Encoder """

    def __init__(self, config):
        super(Encoder, self).__init__()

        n_position = config["max_seq_len"] + 1
        n_src_vocab = len(symbols) + 1
        d_word_vec = config["transformer"]["encoder_hidden"]
        n_layers = config["transformer"]["encoder_layer"]
        n_head = config["transformer"]["encoder_head"]
        d_k = d_v = d_word_vec // n_head
        d_model = d_word_vec
        d_inner = config["transformer"]["conv_filter_size"]
        kernel_size = config["transformer"]["conv_kernel_size"]
        dropout = config["transformer"]["encoder_dropout"]
        self.v_emb_mod = config["mod"]["variational_embedding"]
        self.v_pemb_mod = config["mod"]["variational_phoneme_embedding"]

        self.max_seq_len = config["max_seq_len"]
        self.d_model = d_model

        if self.v_pemb_mod:
            self.src_word_emb_mu = nn.Embedding(n_src_vocab, d_word_vec, padding_idx=Constants.PAD)
            self.src_word_emb_sigma = nn.Embedding(n_src_vocab, d_word_vec, padding_idx=Constants.PAD)
        else:
            self.src_word_emb = nn.Embedding(n_src_vocab, d_word_vec, padding_idx=Constants.PAD)

        if self.v_emb_mod:
            self.mu_layer = nn.Linear(d_model, d_model, bias=True)
            self.sig_layer = nn.Linear(d_model, d_model, bias=True)

        position_enc = get_sinusoid_encoding_table(n_position, d_word_vec).unsqueeze(0)
        self.position_enc = nn.Parameter(position_enc, requires_grad=False)
        layer_stack = [FFTBlock(d_model, n_head, d_k, d_v, d_inner, kernel_size, dropout=dropout) for _ in range(n_layers)]
        self.layer_stack = nn.ModuleList(layer_stack)
    
    def reparameterize(self,mu,sigma,alophone_control=0.01):
        std = torch.exp(0.5*sigma)
        zero = torch.zeros_like(std)
        eps = torch.normal(mean=zero, std=zero+0.5)
        return (mu + alophone_control*eps*std)

    def forward(self, src_seq, mask, alophone_control=0.01):

        batch_size, max_len = src_seq.shape[0], src_seq.shape[1]

        # -- Prepare masks
        slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1)

        # -- Forward
        if self.v_pemb_mod:
            mu = self.src_word_emb_mu(src_seq)
            sigma = self.src_word_emb_sigma(src_seq)
            enc_output = self.reparameterize(mu, sigma, alophone_control)
        else:
            enc_output = self.src_word_emb(src_seq)

        if not self.training and src_seq.shape[1] > self.max_seq_len:
            positionals = get_sinusoid_encoding_table(src_seq.shape[1], self.d_model)
            positionals = positionals[: src_seq.shape[1], :].unsqueeze(0).expand(batch_size, -1, -1)
            positionals = positionals.to(src_seq.device)
        else:
            positionals = self.position_enc[:, :max_len, :].expand(batch_size, -1, -1)
        enc_output = enc_output + positionals

        for enc_layer in self.layer_stack:
            enc_output,_ = enc_layer(enc_output, mask=mask, slf_attn_mask=slf_attn_mask)
        
        if self.v_emb_mod:
            mu = self.mu_layer(enc_output)
            sigma = self.sig_layer(enc_output)
            enc_output = self.reparameterize(mu, sigma, alophone_control)
         
        return enc_output

class Generator(nn.Module):
    """ Formant, Excitation Generator """

    def __init__(self, config, query_projection=False):
        super(Generator, self).__init__()
        self.max_seq_len = config["max_seq_len"]
        n_position = self.max_seq_len + 1
        d_word_vec = d_model = config["transformer"]["encoder_hidden"]
        n_layers = config["transformer"]["generator_layer"]
        n_head = config["transformer"]["encoder_head"]
        d_k = d_v = d_word_vec // n_head
        d_inner = config["transformer"]["conv_filter_size"]
        kernel_size = config["transformer"]["conv_kernel_size"]
        dropout = config["transformer"]["encoder_dropout"]

        self.query_projection = query_projection
        self.d_model = d_model

        position_enc = get_sinusoid_encoding_table(n_position, d_word_vec).unsqueeze(0)
        self.position_enc = nn.Parameter(position_enc, requires_grad=False)

        self.cross_layer = FFTBlock(d_model,n_head,d_k,d_v,d_inner,kernel_size,dropout=dropout,query_projection=query_projection)
        layer_stack = [FFTBlock(d_model,n_head,d_k,d_v,d_inner,kernel_size,dropout=dropout) for _ in range(n_layers - 1)]
        self.layer_stack = nn.ModuleList(layer_stack)

    def forward(self, hidden, mask, hidden_query):
        batch_size, max_len = hidden.shape[0], hidden.shape[1]

        # -- Prepare masks
        slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1)

        # -- Forward
        if not self.training and hidden.shape[1] > self.max_seq_len:
            positionals = get_sinusoid_encoding_table(hidden.shape[1], self.d_model)[: hidden.shape[1], :]
            positionals = positionals.unsqueeze(0).expand(batch_size, -1, -1).to(hidden.device)
        else:
            positionals = self.position_enc[:, :max_len, :].expand(batch_size, -1, -1)
        
        output = hidden + positionals

        output,_ = self.cross_layer(output, mask=mask, slf_attn_mask=slf_attn_mask, hidden_query=hidden_query)
        
        for enc_layer in self.layer_stack:
            output,_ = enc_layer(output, mask=mask, slf_attn_mask=slf_attn_mask)

        return output

class Decoder(nn.Module):
    """ Decoder """

    def __init__(self, config):
        super(Decoder, self).__init__()

        self.max_seq_len = config["max_seq_len"]

        n_position = self.max_seq_len + 1
        d_word_vec = d_model = config["transformer"]["decoder_hidden"]
        n_layers = config["transformer"]["decoder_layer"]
        n_head = config["transformer"]["decoder_head"]
        d_k = d_v = d_word_vec // n_head
        d_inner = config["transformer"]["conv_filter_size"]
        kernel_size = config["transformer"]["conv_kernel_size"]
        dropout = config["transformer"]["decoder_dropout"]

        self.d_model = d_model
        position_enc = get_sinusoid_encoding_table(n_position, d_word_vec).unsqueeze(0)
        self.position_enc = nn.Parameter(position_enc, requires_grad=False)

        layer_stack = [FFTBlock(d_model, n_head, d_k, d_v, d_inner, kernel_size, dropout=dropout) for _ in range(n_layers)]
        self.layer_stack = nn.ModuleList(layer_stack)

    def forward(self, enc_seq, mask):
        batch_size, max_len = enc_seq.shape[0], enc_seq.shape[1]
        # -- Forward
        if not self.training and enc_seq.shape[1] > self.max_seq_len:
            # -- Prepare masks
            slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1)
            # -- Prepare PE
            positionals = get_sinusoid_encoding_table(enc_seq.shape[1], self.d_model)[: enc_seq.shape[1], :]
            positionals = positionals.unsqueeze(0).expand(batch_size, -1, -1).to(enc_seq.device)
            
            dec_output = enc_seq + positionals
        else:
            max_len = min(max_len, self.max_seq_len)
            # -- Prepare masks
            slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1)
            slf_attn_mask = slf_attn_mask[:, :, :max_len]
            # -- Prepare PE
            positionals = self.position_enc[:, :max_len, :].expand(batch_size, -1, -1)
            
            dec_output = enc_seq[:, :max_len, :] + positionals
            mask = mask[:, :max_len]

        for dec_layer in self.layer_stack:
            dec_output,_ = dec_layer(dec_output, mask=mask, slf_attn_mask=slf_attn_mask)

        return dec_output, mask
