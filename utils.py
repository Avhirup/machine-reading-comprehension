import torch 
import pandas as pd 
import numpy as np
import pickle as pkl 
from tqdm import tqdm
from glob import glob

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
		with open(f"{self.data_path}/{index}.pkl","rb") as f:
			d=pkl.load(f).to_dict()
		query=d['Query']
		passages=d['Passages']
		label=d['RelevantPassage']
		passage_tensor=[]
		query_tensor=process_sentence(str(query),self.tokenizer,self.model)
		while len(passages)<10:
			passages.append(" ")
		while len(passages)>10:
			passages.pop()
		for passage in passages:
			passage_tensor.append(process_sentence(str(passage),self.tokenizer,self.model))
		passage_tensor=torch.cat(passage_tensor,0)
		label=torch.LongTensor([label])
		return query_tensor,passage_tensor,label

def process_sentence(sentence,tokenizer,model,MAX_SEQ_LEN=128):
	
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
	try:
		assert len(input_ids) == MAX_SEQ_LEN
		assert len(input_mask) == MAX_SEQ_LEN
		assert len(segment_ids) == MAX_SEQ_LEN

		input_mask=torch.LongTensor([input_mask])
		input_ids=torch.LongTensor([input_ids])
		segment_ids=torch.LongTensor([segment_ids])
	except:
		input_mask=torch.LongTensor([[0]*MAX_SEQ_LEN])
		input_ids=torch.LongTensor([[0]*MAX_SEQ_LEN])
		segment_ids=torch.LongTensor([[0]*MAX_SEQ_LEN])

	all_encoder_layers, pooled_output = model(input_ids,segment_ids,attention_mask=input_mask, output_all_encoded_layers=False)
	return pooled_output

def format(dataframe):
	return pd.DataFrame.from_dict(dict(QID=dataframe['QID'].iloc[0],
		Query=dataframe['Query'].iloc[0],
		Passages=list(map(lambda x:x.encode('utf-8',errors="ignore"),dataframe["Passage"].drop_duplicates().tolist())),
		Relevance=np.array(dataframe["Relevance"].drop_duplicates().tolist()),
		PassageID=np.array(dataframe["PassageID"].drop_duplicates().tolist()),
		RelevantPassage=np.argmax(np.array(dataframe["Relevance"].drop_duplicates().tolist()))),orient='index').T

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

def k(x):
	with open(x,"rb") as f:
		d=pkl.load(f).to_dict()
		if len(d['Passages'])!=10:
			print(d['QID'],len(d['Passages']))
			return False
		else:
			return True



