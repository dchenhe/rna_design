import sys
sys.path.append('/user/hedongcheng/esm')
sys.path.append('/user/hedongcheng/esm/esm')

from util import load_structure
from util import extract_coords_from_structure
fpath ='/user/hedongcheng/inverse_folding/RNA3D_DATA/pdb/1a4d_B.pdb'
fpath2 = '/user/hedongcheng/inverse_folding/RNA3D_DATA/pdb/1a1t_B.pdb'
s = load_structure(fpath)
c, seq = extract_coords_from_structure(s)
s = load_structure(fpath2)
c2, seq2 = extract_coords_from_structure(s)

from esm.data import Alphabet
from util import CoordBatchConverter
dictionary = Alphabet(['A','G','C','U','X'])
batch_converter = CoordBatchConverter(dictionary)
batch = [(c, None, seq),(c2,None,seq2)]
coords, confidence, strs, tokens, padding_mask = batch_converter(
    batch)

from gvp_transformer import GVPTransformerModel
import argparse
# parser = argparse.ArgumentParser(description='Trainning Parameters')
# parser.add_argument('--encoder_embed_dim', dest='encoder_embed_dim', 
#             type=int, default = 512, help=' embedding dimension ')
# parser.add_argument('--decoder_embed_dim', dest='decoder_embed_dim', 
#             type=int, default = 512, help=' embedding dimension ')
# args = parser.parse_args()
class A:
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
        
args = A(512,512,0.1)


model = GVPTransformerModel(args, dictionary)

prev_output_tokens = tokens[:, :-1]
target = tokens[:, 1:]
target_padding_mask = (target == dictionary.padding_idx)
c = coords[:,:,[0,1,2,3],:]
adc = coords[:,:,[0,1,2,4,5,6,7,8],:]
logits,_ = model.forward(c,adc,padding_mask.bool(),confidence,prev_output_tokens)