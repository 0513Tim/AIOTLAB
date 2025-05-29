import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float()*(-math.log(10000.0)/d_model))
        pe[:,0::2] = torch.sin(pos*div)
        pe[:,1::2] = torch.cos(pos*div)
        self.register_buffer("pe", pe.unsqueeze(0))   # (1,L,d)
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class TimeSeriesTFM(nn.Module):
    def __init__(self, input_dim=3, output_dim=3,
                 d_model=64, nhead=4,
                 num_layers=3, dim_feedforward=128, dropout=0.1):
        super().__init__()
        self.inp = nn.Linear(input_dim, d_model)
        self.pos = PositionalEncoding(d_model)
        self.tfm = nn.Transformer(d_model=d_model, nhead=nhead,
                                  num_encoder_layers=num_layers,
                                  num_decoder_layers=num_layers,
                                  dim_feedforward=dim_feedforward,
                                  dropout=dropout,
                                  batch_first=True)
        self.out = nn.Linear(d_model, output_dim)
    def forward(self, src, tgt):          # src:(B,1344,3)  tgt:(B,168,3)
        src = self.pos(self.inp(src))
        tgt = self.pos(self.inp(tgt))
        out = self.tfm(src, tgt)
        return self.out(out)              # (B,168,3)
