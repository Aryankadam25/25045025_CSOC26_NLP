import torch
import torch.nn as nn 
import torch.nn.functional as F


class CausalAttention(nn.Module):
    def __init__(self,n_embedding,n_heads,block_size,dropout):
        super().__init__()
        self.n_heads=n_heads
        self.head_size=n_embedding//n_heads
        
        self.key=nn.Linear(n_embedding,n_embedding,bias=False)
        self.query=nn.Linear(n_embedding,n_embedding,bias=False)
        self.value=nn.Linear(n_embedding,n_embedding,bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        
        self.proj= nn.Linear(n_embedding,n_embedding)
        self.dropout=nn.Dropout(dropout)
    
    def forward(self,X):
        B,T,C=X.shape
        k=self.key(X) #(B, T, C)
        q=self.query(X) #(B, T, C)
        v=self.value(X) #(B, T, C)
        
        k = k.view(B, T, self.n_heads, self.head_size).transpose(1, 2) # (B,n_heads,T,head_size)
        q = q.view(B, T, self.n_heads, self.head_size).transpose(1, 2) # (B,n_heads,T,head_size)
        v = v.view(B, T, self.n_heads, self.head_size).transpose(1, 2) # (B,n_heads,T,head_size)
        
        wei = (q @ k.transpose(-2, -1)) * (self.head_size ** -0.5) # (B, n_heads, T, T)
        wei=wei.masked_fill(self.tril[:T,:T]==0 , float("-inf"))
        wei=self.dropout(F.softmax(wei,dim=-1))
        
        output=wei@v #(B, n_heads, T, head_size)
        output = output.transpose(1, 2).reshape(B, T, C)
        output = self.proj(output)
        output = self.dropout(output)
        return output
    
    
class FeedForwardNetwork(nn.Module):
    def __init__(self,n_embedding,dropout):
        super().__init__()
        self.forw=nn.Sequential(
            nn.Linear(n_embedding,4*n_embedding),
            nn.GELU(),
            nn.Linear(4*n_embedding,n_embedding),
            nn.Dropout(dropout)
        )
    def forward(self,X):
        return self.forw(X)
   
        
class TransformerBlock(nn.Module):
    def __init__(self,n_embedding,n_heads,block_size,dropout):
        super().__init__()
        self.ln1=nn.LayerNorm(n_embedding)
        self.ln2=nn.LayerNorm(n_embedding)
        self.SelfAttention=CausalAttention(n_embedding,n_heads,block_size,dropout)
        self.ffn=FeedForwardNetwork(n_embedding,dropout)
        
    def forward(self,X):
        X=X+self.SelfAttention(self.ln1(X))   
        X=X+self.ffn(self.ln2(X))
        return X

class DecoderTransformer(nn.Module):
    def __init__(self,vocab_size,n_embedding,n_heads,block_size,dropout,n_layers):
        super().__init__()
        self.block_size=block_size
        self.token_embedding=nn.Embedding(vocab_size,n_embedding)
        self.positional_embedding=nn.Embedding(block_size,n_embedding)
        self.emb_dropout=nn.Dropout(dropout)
        self.block=nn.Sequential(*[
            TransformerBlock(n_embedding,n_heads,block_size,dropout) for _ in range(n_layers)
            ])
        self.ln=nn.LayerNorm(n_embedding)
        # tying weight of the final linear layer to the token_embedding layer to reduce parameter bloat and
        # as we know that nn.Linear(a,b) stores the weights in the shape of (b,a) thats why directly equating it
        # to the weights of token_embeddings
        self.lm_head=nn.Linear(n_embedding,vocab_size,bias=False)
        self.lm_head.weight=self.token_embedding.weight
        
        self.apply(self._init_weights)
        
    def _init_weights(self,module):
        if isinstance(module,nn.Linear):
            torch.nn.init.normal_(module.weight,mean=0.0,std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)   
        elif isinstance(module,nn.Embedding):
            torch.nn.init.normal_(module.weight,mean=0.0,std=0.02)
    
    def forward(self,X):
        # dim x = (B,T)
        B,T=X.shape
        assert T <=self.block_size,f"maximum length a sequence can have is {self.block_size}"
        pos=torch.arange(0,T,dtype=torch.long,device=X.device)
        tok_emb=self.token_embedding(X) # (B,T,n_embedding)
        pos_emb=self.positional_embedding(pos) # (T,n_embedding)
        pos_emb=pos_emb.unsqueeze(0) #(1,T,n_embedding)
        X=self.emb_dropout(tok_emb+pos_emb) # (B,T,n_embedding)
        X=self.block(X) # (B,T,n_embedding)
        X=self.ln(X) # (B,T,n_embedding)
        logits=self.lm_head(X) #(B,T,vocab_size)
        return logits
    