import h5py
from utils import dump_row
import pickle as pkl

with open("../data/data_grouped.pkl","rb") as f:
	data=pkl.load(f)

dump_row(data) 