import torch 
import pandas as pd 
import numpy as np
import pickle as pkl 
from tqdm import tqdm
from glob import glob
import gc
import h5py

from torch.utils.data import Dataset 

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertModel,BertConfig
SENTENCE_COUNTER=0
class MRCDataset(Dataset):
	"""docstring for Dataset"""
	def __init__(self, data_path,model_type="bert-base-uncased"):
		super(MRCDataset, self).__init__()
		self.data_path=data_path
		self.data_files=glob(data_path+"/*")
		self.model_type = model_type
		self.model=BertModel.from_pretrained(model_type)
		self.tokenizer=BertTokenizer.from_pretrained(model_type)

	def __len__(self):
		return len(self.data_files)

	def __getitem__(self,index):
		#traverse to the file
		with h5py.File(f"{self.data_path}/{index}.hdf5","r") as f:
			query=f['query'].value
			passages=f['passages'].value
			label=f['label'].value
		return query,passages,label

def process_sentence(sentence,tokenizer,model,MAX_SEQ_LEN=128,is_cuda=True):
	
	tokens_sent=tokenizer.tokenize(sentence)
	tokens = []
	segment_ids = []
	tokens.append("[CLS]")
	segment_ids.append(0)
	for token in tokens_sent:
		tokens.append(token)
		segment_ids.append(0)
	tokens.append("[SEP]")
	segment_ids.append(0)

	input_ids=tokenizer.convert_tokens_to_ids(tokens)
	input_mask=[1] * len(input_ids)

	#ZeroPadding
	while len(input_ids)<MAX_SEQ_LEN:
		input_ids.append(0)
		input_mask.append(0)
		segment_ids.append(0)
	while len(input_ids)>MAX_SEQ_LEN:
		input_ids.pop()
		input_mask.pop()
		segment_ids.pop()
	assert len(input_ids) == MAX_SEQ_LEN
	assert len(input_mask) == MAX_SEQ_LEN
	assert len(segment_ids) == MAX_SEQ_LEN

	input_mask=torch.LongTensor([input_mask])
	input_ids=torch.LongTensor([input_ids])
	segment_ids=torch.LongTensor([segment_ids])
	if is_cuda:
		input_ids,input_mask,segment_ids=input_ids.cuda(),input_mask.cuda(),segment_ids.cuda()
	all_encoder_layers, pooled_output = model(input_ids,segment_ids,attention_mask=input_mask, output_all_encoded_layers=False)
	return pooled_output

def format(dataframe):
	return pd.DataFrame.from_dict(dict(QID=dataframe['QID'].iloc[0],
		Query=dataframe['Query'].iloc[0],
		Passages=list(map(lambda x:x.encode('utf-8',errors="ignore"),dataframe["Passage"].tolist())),
		Relevance=np.array(dataframe["Relevance"].tolist()),
		PassageID=np.array(dataframe["PassageID"].tolist()),
		RelevantPassage=np.argmax(np.array(dataframe["Relevance"].tolist()))),orient='index').T

def group_query_passage(dataframe):
	return dataframe.groupby("QID",as_index=False).apply(lambda x:format(x)).reset_index(drop=True)


def dump_dataframe(data,dump_path):
	for index in tqdm(data.index):
		_id=data['QID'].iloc[index]
		with open(dump_path+f"/{_id}.pkl","wb") as f:
			pkl.dump(data.iloc[index,:],f)	

def convert_data_to_rows(data_path="../data/data.tsv",dump_path="../data/data_rows/"):
	data=pd.read_csv(data_path,header=None,sep="\t",names=['QID',"Query","Passage","Relevance","PassageID"])
	print(111)
	data=group_query_passage(data)
	print(111)
	dump_dataframe(data,dump_path)

def dump_row(data,dump_path="../data/data_h5py/"):
	model_type="bert-base-uncased"
	model=BertModel.from_pretrained(model_type).cuda()
	tokenizer=BertTokenizer.from_pretrained(model_type)
	for index in tqdm(data.index):
		_id=data['QID'].iloc[index]
		d=data.iloc[index].to_dict()
		query=d['Query']
		passages=d['Passages']
		label=d['RelevantPassage']
		query_tensor=process_sentence(str(query),tokenizer,model).cpu().detach().numpy()
		passage_tensor=[]
		for passage in passages:
			passage_tensor.append(process_sentence(str(passage),tokenizer,model))
		passage_tensor=torch.cat(passage_tensor,0).cpu().detach().numpy()
		label=torch.LongTensor([label]).numpy()

		data_dict=dict(query=query_tensor,passages=passage_tensor,label=label)
		with h5py.File(f'{dump_path}/{_id}.hdf5','w') as h: 
			for k, v in data_dict.items():
				h.create_dataset(k, data=v)
		del data_dict
		del passage_tensor
		del query_tensor
		gc.collect()


def k(x):
	with open(x,"rb") as f:
		d=pkl.load(f).to_dict()
		if len(d['Passages'])!=10:
			print(d['QID'],len(d['Passages']))
			return False
		else:
			return True



