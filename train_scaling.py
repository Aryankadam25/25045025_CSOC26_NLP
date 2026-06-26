import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import urllib.request
from models import DecoderTransformer

def prepare_data():
    if not os.path.exists('input.txt'):
        url="https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        urllib.request.urlretrieve(url,'input.txt')
    with open('input.txt','r',encoding="utf-8") as f:
        text=f.read()
    
    chars=sorted(list(set(text)))
    vocab_size=len(chars)
    stoi={ch:i for i,ch in enumerate(chars)}
    itos={i:ch for ch ,i in stoi.items()}

    encoder=lambda i : [stoi[j] for j in i]
    decoder=lambda i : [itos[j] for j in i]

    data=torch.tensor(encoder(text),dtype=torch.long)
    n=int(0.9*len(data))
    train_data=data[:n]
    val_data=data[n:]
    
    return train_data , val_data , vocab_size

block_size=256
n_embeddings=312
n_head=6
n_layers=6
dropout=0.2
def get_batch(data,block_size,batch_size,device):
    ix=torch.randint(len(data)-block_size,(batch_size,))
    X=torch.stack([data[i:i+block_size] for i in ix])
    y=torch.stack([data[i+1:i+block_size+1] for i in ix])
    X=X.to(device)
    y=y.to(device)
    return X , y

def count_non_embedding_params(model):
    return sum(p.numel() for name, p in model.named_parameters() 
               if 'embed' not in name and 'lm_head' not in name)

@torch.no_grad()
def estimate_loss(model ,train_data,val_data,block_size,batch_size,device,iterations=100):
    model.eval()
    out={}
    for name,data in [('train',train_data),('val',val_data)]:
        losses=torch.zeros(iterations)
        for i in range(iterations):
            X,y=get_batch(data,block_size,batch_size,device)
            logits=model(X)
            B,T,C=logits.shape
            loss=F.cross_entropy(logits.view(B*T,C),y.view(B*T))
            losses[i]=loss.item()
        out[name]=losses.mean().item()
    model.train()
    return out

def training_sweeps(model,train_data,val_data,block_size,batch_size,device,max_steps=3000):
    model.to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=5e-4,weight_decay=0.1)
    best_val_loss=float("inf")
    for step in range(max_steps):
        xb,yb=get_batch(train_data,block_size,batch_size,device)
        logits=model(xb)
        B,T,C=logits.shape
        loss=F.cross_entropy(logits.view(B*T,C),yb.view(B*T))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(),1.0)
        optimizer.step()
        
        if step%500==0 or step == max_steps-1:
            losses=estimate_loss(model,train_data,val_data,block_size,batch_size,device)
            if losses['val'] < best_val_loss:
                best_val_loss=losses['val']
    print(f"Best Validation Loss (L): {best_val_loss}")
    return best_val_loss

if __name__== "__main__":
    device="cuda" if torch.cuda.is_available() else "cpu"
    train_data, val_data, vocab_size = prepare_data()
    batch_size=64
    sweep_configs = {
        "Tiny":   {"n_layers": 2, "n_embedding": 64,  "n_heads": 2, "block_size": 128},
        "Small":  {"n_layers": 4, "n_embedding": 128, "n_heads": 4, "block_size": 256},
        "Medium": {"n_layers": 6, "n_embedding": 256, "n_heads": 8, "block_size": 256}
    }
    
    scaling_results = {}
    for name , config in sweep_configs.items():
        model=DecoderTransformer(vocab_size=vocab_size,
                                n_embedding=config["n_embedding"],
                                n_heads=config["n_heads"],
                                block_size=config["block_size"],
                                dropout=0.2,
                                n_layers=config["n_layers"])
        n_params=count_non_embedding_params(model)
        print(f"Parameter Count (N) for {name}: {n_params}")
        best_loss =training_sweeps(
            model=model,
            train_data=train_data,
            val_data=val_data,
            block_size=config["block_size"],
            batch_size=batch_size,
            device=device,
            max_steps=3000
        )
        scaling_results[name] = {"N": n_params, "L": best_loss}
        
        for model_name, metrics in scaling_results.items():
            print(f"{model_name} | Parameters (N): {metrics['N']} | Best Val Loss (L): {metrics['L']}")
            
            
    data_fractions=[0.10,0.25,0.50,1.00]
    data_scaling_result={}
    small_config=sweep_configs['Small']
    for frac in data_fractions:
        subset_size=int(frac*len(train_data))
        subset_train_data=train_data[:subset_size]
        model=DecoderTransformer(vocab_size=vocab_size,
                                n_embedding=small_config["n_embedding"],
                                n_heads=small_config["n_heads"],
                                block_size=small_config["block_size"],
                                dropout=0.2,
                                n_layers=small_config["n_layers"])
        best_loss=training_sweeps(model=model,
                                  train_data=subset_train_data,
                                  val_data=val_data,
                                  block_size=small_config['block_size'],
                                  batch_size=batch_size,
                                  device=device,
                                  max_steps=3000)
        data_scaling_result[f'{int(frac*100)}%']={'D':subset_size,'L':best_loss}
        for name, metrics in data_scaling_result.items():
            print(f"{name} | D: {metrics['D']} | Best L: {metrics['L']}")