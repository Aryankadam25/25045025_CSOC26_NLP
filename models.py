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
    
# Testing   
if __name__ == "__main__":
    B,T,C=4,32,64
    vocab_size=100
    model = DecoderTransformer(vocab_size=vocab_size,
                                   n_embedding=C,
                                   n_heads=4,
                                   block_size=T,
                                   dropout=0.0,
                                   n_layers=3)
    x = torch.randint(0, vocab_size, (B, T))
    pos = torch.arange(0, T, dtype=torch.long)
    tok_emb = model.token_embedding(x)
    pos_emb = model.positional_embedding(pos)
    current_x = tok_emb + pos_emb
    assert current_x.shape == (B, T, C), " Embedding dimension mismatch!"
    
    # verifying every transformer block
    for i , block in enumerate(model.block):
        current_x = block(current_x)
        assert current_x.shape == (B, T, C), f" Dimension mismatch at Block {i}!"
        
    # Verify Final LayerNorm and Output Head
    current_x = model.ln(current_x)
    logits = model.lm_head(current_x)
    assert logits.shape == (B, T, vocab_size), "Final output (lm_head) dimension mismatch!"
    
    # Testing Causal Compliance (No future leakage)
    x_embed = torch.randn(1, 10, C, requires_grad=True)
    out = x_embed
    for block in model.block:
        out = block(out)
    loss = out[0, 5, :].sum()# Take gradient of the 5th token
    loss.backward()
    
    assert torch.all(x_embed.grad[0, 6:, :] == 0), "Causal Leak Detected! Future tokens affecting past/present."
    
    #Testing Single Batch Overfitting to ZERO loss
    model_overfit = DecoderTransformer(vocab_size=vocab_size,
                                       n_embedding=C,
                                       n_heads=4,
                                       block_size=16,
                                       dropout=0.0,
                                       n_layers=2)
    optimizer = torch.optim.AdamW(model_overfit.parameters(), lr=0.01)
    x_batch = torch.randint(0, vocab_size, (1, 16))
    y_batch = torch.randint(0, vocab_size, (1, 16))
    loss_val = float('inf')
    step = 0
    while loss_val > 5e-4 and step < 500:
        logits = model_overfit(x_batch)
        loss = F.cross_entropy(logits.view(-1, vocab_size), y_batch.view(-1))
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
        loss_val = loss.item()
        step += 1
        
    assert loss_val <= 5e-4, f"Failed to overfit. Final loss stuck at: {loss_val}"
    print("no errors")