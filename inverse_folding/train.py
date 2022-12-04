import sys
sys.path.append('/user/hedongcheng/esm')
sys.path.append('/user/hedongcheng/esm/esm')
from gvp_transformer import GVPTransformerModel
from esm.data import Alphabet
import torch
import torch.nn.functional as F
from tqdm import tqdm
import torch.optim as optim
import numpy as np

class args_class: # use the same param as esm-if1, waiting to be adjusted...
    def __init__(self,encoder_embed_dim,decoder_embed_dim,dropout):
        self.encoder_embed_dim = encoder_embed_dim
        self.decoder_embed_dim = decoder_embed_dim
        self.dropout = dropout
        self.gvp_top_k_neighbors=30
        self.gvp_node_hidden_dim_vector=256
        self.gvp_node_hidden_dim_scalar=1024
        self.gvp_edge_hidden_dim_scalar=32
        self.gvp_edge_hidden_dim_vector=1
        self.gvp_num_encoder_layers=4
        self.gvp_dropout=0.1
        self.encoder_layers=8
        self.encoder_attention_heads=8
        self.attention_dropout=0.1
        self.encoder_ffn_embed_dim=2048
        self.decoder_layers=8
        self.decoder_attention_heads=8
        self.decoder_ffn_embed_dim=2048
        
args = args_class(512,512,0.1)

dictionary = Alphabet(['A','G','C','U','X'])
model = GVPTransformerModel(args, dictionary)

def train(model,epochs,lr,dataloader):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in tqdm(range(epochs)):
        ls = []
        model.train()
        for coords, padding_mask, confidence, tokens in dataloader:
            prev_output_tokens = tokens[:, :-1]
            target = tokens[:, 1:]
            c = coords[:,:,[0,1,2,3],:] # the four backbone atoms
            adc = coords[:,:,[0,1,2,4,5,6,7,8],:] # eight atoms which is used to compute dihedral angles
            logits,_ = model.forward(c,adc,padding_mask.bool(),confidence,prev_output_tokens)
            loss = F.cross_entropy(logits, target, reduction='none')
            loss.backward()
            loss = loss[0].cpu().detach().numpy()
            ls.append(sum(loss))
            optimizer.step()
            optimizer.zero_grad()
        ls = np.mean(ls)
        print(f'Training loss {epoch}: {ls}')
        torch.save(model.state_dict(),'/user/hedongcheng/esm/inverse_folding/saved_model/model_'+str(epoch)+'.pth')
        
        
        
