import torch 
import torch.nn as nn 
import math 

class InputEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super(InputEmbedding, self).__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)


    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int = 512, seq_len: int = 5000, dropout: float = 0.1):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model 
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(d_model, seq_len)
        pos = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000) / d_model))

        pe[:, 0::2] = torch.sin(pos * div_term) # Even indices
        pe[:, 1::2] = torch.cos(pos * div_term) # Odd indices


        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.shape[1], :].requires_grad_(False)
        return self.dropout(x)

class MultiHeadAttention(nn.Module):

    def __init__(self, d_model: int, h: int, dropout: float = 0.1): 
        super.__init__(MultiHeadAttention, self).__init__()
        
        assert d_model % h == 0, "d_model must be divisible by h"
        self.d_model = d_model # Size of embedding vectors
        self.h = h # Number of heads 
            
        self.d_k = d_model // h   
        self.w_k = nn.Linear(d_model, d_model, bias=False) # Wk 
        self.w_q = nn.Linear(d_model, d_model, bias=False) # Wq 
        self.w_v = nn.Linear(d_model, d_model, bias=False) # Wv 
        self.w_o = nn.Linear(d_model, d_model, bias=False) # Wo 
        self.dropout = nn.Dropout(dropout)


    @staticmethod
    def scaled_dot_product_attention(query, key, value, mask, dropout=nn.Dropout):

        d_k = key.size(-1)

        scores =  (query @ key.transpose(-2, -1)) / math.sqrt(d_k) # Attention scores
        if mask is not None: 
            scores.masked_fill(mask == 0, -1e9)
        if dropout is not None:
            weights = dropout(weights)

        return weights @ value, weights

    def forward(self, q, k, v, mask):
        batch_size=  q.shape[0]

        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_k) -> (batch, h, seq_len, d_k)
        query = query.view(batch_size, -1, self.h, self.d_k).transpose(1, 2)  # (batch_size, h, seq_len, d_k)
        key = key.view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        value = value.view(batch_size, -1, self.h, self.d_k).transpose(1, 2)

        x, scores = MultiHeadAttention.scaled_dot_product_attention(query, key, value, self.dropout)

        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_moel)
        return self.w_o(x)


class FeedForwardNet(nn.Module):

    super.__init__()
    def __init__(self, d_model: int, d_ff: int, dropout: flot):
        super(FeedForwardNet, self).__init__()
        self.d_model = d_model 
        self.d_ff = d_ff
        self.dropout = nn.Dropout(dropout)

        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout) 
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):

        x = self.linear_1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = linear_2(x)
        return x 



if __name__ == "__main__":

    vocab_size = 1000
    d_model = 512
    seq_len = 100
    h = 8
    d_ff = 2048
    dropout = 0.1 

    x = torch.randint(0, vocab_size, (2, 10))

    # Test InputEmbedding
    input_emb = InputEmbedding(vocab_size, d_model)
    emb_output = input_emb(x)
    print("InputEmbedding output shape:", emb_output.shape)

    # Test PositionalEncoding
    pos_enc = PositionalEncoding(d_model, max_len, dropout)
    pos_output = pos_enc(emb_output)
    print("PositionalEncoding output shape:", pos_output.shape)

    # Test MultiHeadAttention
    mha = MultiHeadAttention(d_model, h, dropout)
    mha_output = mha(pos_output, pos_output, pos_output)
    print("MultiHeadAttention output shape:", mha_output.shape)

    # Test FeedForwardNet
    ffn = FeedForwardNet(d_model, d_ff, dropout)
    ffn_output = ffn(mha_output)
    print("FeedForwardNet output shape:", ffn_output.shape)










