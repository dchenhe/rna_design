import sys
sys.path.append('/user/hedongcheng/esm')
sys.path.append('/user/hedongcheng/esm/esm')
from esm.data import Alphabet
from util import load_structure
from util import get_atom_coords_residuewise
from util import CoordBatchConverter
from gvp_transformer import GVPTransformerModel
import argparse

fpath ='/user/hedongcheng/data_20210717_20220312/native_out_20210717_20220312/2021-07-17_00000021_1_native.pdb'
## get coords
s = load_structure(fpath)
coords = get_atom_coords_residuewise(['N','CA','C'], s)

dictionary = Alphabet(['A','G','C','U','X'])
batchconverter = CoordBatchConverter(dictionary)
batch_coords, confidence, _, _, padding_mask = (
    batchconverter([(coords,None,None)])
)

parser = argparse.ArgumentParser(description='Trainning Parameters')
parser.add_argument('--encoder_embed_dim', dest='encoder_embed_dim', 
            type=int, default = 512, help=' embedding dimension ')
parser.add_argument('--decoder_embed_dim', dest='decoder_embed_dim', 
            type=int, default = 512, help=' embedding dimension ')
args = parser.parse_args()

model = GVPTransformerModel(args, dictionary)
logits, extra = model(coords,padding_mask,confidence)



    