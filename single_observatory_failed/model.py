# model.py
import math, torch
import torch.nn as nn

# ---------- Positional Encoding ----------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, L, d)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

# ---------- Transformer for Time-series ----------
class TimeSeriesTFM(nn.Module):
    """
    If teacher_force=True  ➜  常規 forward，用真值當 decoder input
    If teacher_force=False ➜  Autoregressive generate，不吃真值
    """
    def __init__(self,
                 input_dim=3, output_dim=3,
                 d_model=64, nhead=4,
                 num_layers=3, dim_feedforward=128,
                 dropout=0.1, max_len=2000):
        super().__init__()
        self.pred_len = 168                      # 預測步長（可改）
        self.inp  = nn.Linear(input_dim, d_model)
        self.pos  = PositionalEncoding(d_model, max_len=max_len)
        self.tfm  = nn.Transformer(
            d_model=d_model, nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.out = nn.Linear(d_model, output_dim)

    # ----------- Teacher-forcing Forward -----------
    def forward(self, src, tgt=None, *, teacher_force=True):
        """
        src : (B, L_enc, input_dim)
        tgt : (B, L_dec, input_dim), True values if teacher_force
        """
        if teacher_force:
            assert tgt is not None, "teacher_force=True 時必須提供 tgt"
            src = self.pos(self.inp(src))        # (B, L_enc, d)
            tgt_emb = self.pos(self.inp(tgt))    # (B, L_dec, d)
            out = self.tfm(src, tgt_emb)         # (B, L_dec, d)
            return self.out(out)                 # (B, L_dec, output_dim)
        else:
            # -------- Autoregressive generate --------
            return self.generate(src, horizon=tgt.shape[1] if tgt is not None
                                                else self.pred_len)

    # ----------- Greedy Generate -----------
    @torch.no_grad()
    def generate(self, src, *, horizon=None, start_token=None):
        """
        src       : (B, L_enc, input_dim)
        horizon   : 預測長度（int）
        start_token: (B, 1, input_dim) or None → 用最後一個 encoder step
        return    : (B, horizon, output_dim)
        """
        if horizon is None:
            horizon = self.pred_len
        B, _, D = src.shape
        device  = src.device

        # (1) encode once
        memory = self.tfm.encoder(self.pos(self.inp(src)))

        # (2) 視情況選 BOS token (=上一時刻真值或 0)
        if start_token is None:
            start_token = src[:, -1:, :]         # (B, 1, D_in)
        dec_input = self.pos(self.inp(start_token))  # (B, 1, d_model)

        preds = []
        for _ in range(horizon):
            out_step = self.tfm.decoder(dec_input, memory)[:, -1:, :]  # last step
            y_step  = self.out(out_step)            # (B, 1, out_dim)
            preds.append(y_step)

            # 把上一預測嵌入，接到 decoder input
            y_emb = self.pos(self.inp(y_step))
            dec_input = torch.cat([dec_input, y_emb], dim=1)   # 增長序列

        return torch.cat(preds, dim=1)              # (B, horizon, out_dim)
