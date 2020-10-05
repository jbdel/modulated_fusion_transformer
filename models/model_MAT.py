import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.fc import MLP, FC
from layers.layer_norm import LayerNorm

# ------------------------------------
# ---------- Masking sequence --------
# ------------------------------------
def make_mask(feature):
    return (torch.sum(
        torch.abs(feature),
        dim=-1
    ) == 0).unsqueeze(1).unsqueeze(2)



class AttFlat(nn.Module):
    def __init__(self, args):
        super(AttFlat, self).__init__()
        self.args = args
        self.flat_glimpse = 1

        self.mlp = MLP(
            in_size=args.hidden_size,
            mid_size=args.ff_size,
            out_size=self.flat_glimpse,
            dropout_r=args.dropout_r,
            use_relu=True
        )

        self.linear_merge = nn.Linear(
            args.hidden_size * self.flat_glimpse,
            args.hidden_size * 2
        )

    def forward(self, x, x_mask):
        att = self.mlp(x)
        att = att.masked_fill(
            x_mask.squeeze(1).squeeze(1).unsqueeze(2),
            -1e9
        )
        att = F.softmax(att, dim=1)

        att_list = []
        for i in range(self.flat_glimpse):
            att_list.append(
                torch.sum(att[:, :, i: i + 1] * x, dim=1)
            )

        x_atted = torch.cat(att_list, dim=1)
        x_atted = self.linear_merge(x_atted)

        return x_atted



class MHAtt(nn.Module):
    def __init__(self, args):
        super(MHAtt, self).__init__()
        self.args = args

        self.linear_v = nn.Linear(args.hidden_size, args.hidden_size)
        self.linear_k = nn.Linear(args.hidden_size, args.hidden_size)
        self.linear_q = nn.Linear(args.hidden_size, args.hidden_size)
        self.linear_merge = nn.Linear(args.hidden_size, args.hidden_size)

        self.dropout = nn.Dropout(args.dropout_r)

    def forward(self, v, k, q, mask):
        n_batches = q.size(0)

        v = self.linear_v(v).view(
            n_batches,
            -1,
            self.args.multi_head,
            int(self.args.hidden_size / self.args.multi_head)
        ).transpose(1, 2)

        k = self.linear_k(k).view(
            n_batches,
            -1,
            self.args.multi_head,
            int(self.args.hidden_size / self.args.multi_head)
        ).transpose(1, 2)

        q = self.linear_q(q).view(
            n_batches,
            -1,
            self.args.multi_head,
            int(self.args.hidden_size / self.args.multi_head)
        ).transpose(1, 2)

        atted = self.att(v, k, q, mask)
        atted = atted.transpose(1, 2).contiguous().view(
            n_batches,
            -1,
            self.args.hidden_size
        )

        atted = self.linear_merge(atted)

        return atted

    def att(self, value, key, query, mask):
        d_k = query.size(-1)

        scores = torch.matmul(
            query, key.transpose(-2, -1)
        ) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)

        att_map = F.softmax(scores, dim=-1)
        att_map = self.dropout(att_map)

        return torch.matmul(att_map, value)


# ---------------------------
# ---- Feed Forward Nets ----
# ---------------------------

class FFN(nn.Module):
    def __init__(self, args):
        super(FFN, self).__init__()

        self.mlp = MLP(
            in_size=args.hidden_size,
            mid_size=args.ff_size,
            out_size=args.hidden_size,
            dropout_r=args.dropout_r,
            use_relu=True
        )

    def forward(self, x):
        return self.mlp(x)


# ------------------------
# ---- Self Attention ----
# ------------------------

class SA(nn.Module):
    def __init__(self, args):
        super(SA, self).__init__()

        self.mhatt = MHAtt(args)
        self.ffn = FFN(args)

        self.dropout1 = nn.Dropout(args.dropout_r)
        self.norm1 = LayerNorm(args.hidden_size)

        self.dropout2 = nn.Dropout(args.dropout_r)
        self.norm2 = LayerNorm(args.hidden_size)

    def forward(self, y, y_mask):
        y = self.norm1(y + self.dropout1(
            self.mhatt(y, y, y, y_mask)
        ))

        y = self.norm2(y + self.dropout2(
            self.ffn(y)
        ))

        return y


# -------------------------------
# ---- Self Guided Attention ----
# -------------------------------

class SGA(nn.Module):
    def __init__(self, args):
        super(SGA, self).__init__()

        self.mhatt1 = MHAtt(args)
        self.mhatt2 = MHAtt(args)
        self.ffn = FFN(args)

        self.dropout1 = nn.Dropout(args.dropout_r)
        self.norm1 = LayerNorm(args.hidden_size)

        self.dropout2 = nn.Dropout(args.dropout_r)
        self.norm2 = LayerNorm(args.hidden_size)

        self.dropout3 = nn.Dropout(args.dropout_r)
        self.norm3 = LayerNorm(args.hidden_size)

    def forward(self, x, y, x_mask, y_mask):
        x = self.norm1(x + self.dropout1(
            self.mhatt1(v=x, k=x, q=x, mask=x_mask)
        ))

        x = self.norm2(x + self.dropout2(
            self.mhatt2(v=y, k=y, q=x, mask=y_mask)
        ))

        x = self.norm3(x + self.dropout3(
            self.ffn(x)
        ))

        return x


# ------------------------------------------------
# ---- MAC Layers Cascaded by Encoder-Decoder ----
# ------------------------------------------------

class MCA_ED(nn.Module):
    def __init__(self, args):
        super(MCA_ED, self).__init__()

        self.enc_list = nn.ModuleList([SA(args) for _ in range(args.layer)])
        self.dec_list = nn.ModuleList([SGA(args) for _ in range(args.layer)])

    def forward(self, y, x, y_mask, x_mask):
        # Get encoder last hidden vector
        for enc in self.enc_list:
            y = enc(y, y_mask)

        # Input encoder last hidden vector
        # And obtain decoder last hidden vectors
        for dec in self.dec_list:
            x = dec(x, y, x_mask, y_mask)

        return y, x



# -------------------------
# ---- Main MCAN Model ----
# -------------------------

class Model_MAT(nn.Module):
    def __init__(self, args, vocab_size, pretrained_emb):
        super(Model_MAT, self).__init__()
        self.args = args

        # LSTM
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=args.word_embed_size
        )

        # Loading the GloVe embedding weights
        self.embedding.weight.data.copy_(torch.from_numpy(pretrained_emb))
        self.input_drop = nn.Dropout(args.dropout_i)

        self.lstm_x = nn.LSTM(
            input_size=args.word_embed_size,
            hidden_size=args.hidden_size,
            num_layers=1,
            batch_first=True
        )

        self.lstm_y = nn.LSTM(
            input_size=args.audio_feat_size,
            hidden_size=args.hidden_size,
            num_layers=1,
            batch_first=True
        )

        # self.adapter = nn.Linear(args.audio_feat_size, args.hidden_size)
        self.backbone = MCA_ED(args)

        # Flatten to vector
        self.attflat_img = AttFlat(args)
        self.attflat_lang = AttFlat(args)

        # Classification layers

        self.proj_norm = LayerNorm(2 * args.hidden_size)
        self.proj = nn.Linear(2 * args.hidden_size, args.ans_size)
        self.proj_drop = nn.Dropout(args.dropout_o)

    def forward(self, x, y, _):
        x_mask = make_mask(x.unsqueeze(2))
        y_mask = make_mask(y)

        embedding = self.embedding(x)

        x, _ = self.lstm_x(self.input_drop(embedding))
        y, _ = self.lstm_y(self.input_drop(y))

        # Backbone Framework
        lang_feat, img_feat = self.backbone(
            x,
            y,
            x_mask,
            y_mask
        )

        # Flatten to vector
        lang_feat = self.attflat_lang(
            lang_feat,
            x_mask
        )

        img_feat = self.attflat_img(
            img_feat,
            y_mask
        )

        # Classification layers
        proj_feat = lang_feat + img_feat
        proj_feat = self.proj_norm(proj_feat)

        proj_feat = self.proj_drop(proj_feat)
        proj_feat = self.proj(proj_feat)

        return proj_feat