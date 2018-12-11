"""Main File to """
import os
import argparse
import random
import logging
from tqdm import tqdm,trange

import numpy as np 
import pandas as pd 

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler,SubsetRandomSampler
from torch.utils.data.distributed import DistributedSampler
import torch.optim as optim

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertForSequenceClassification
from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE

from utils import MRCDataset
from model import Model as SimpleCNNModel

from fastai import *


logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s', 
					datefmt = '%m/%d/%Y %H:%M:%S',
					level = logging.INFO)
logger = logging.getLogger(__name__)

def train(model,optimizer,train_dl,criterion,epoch,is_cuda=True):
	running_loss = 0.0
	model.train()
	if next(model.parameters()).is_cuda:
		print("On cuda")
	else:
		print("not on cuda")
	for i, data in enumerate(train_dl, 0):
		# get the inputs
		(query_tensor,passage_tensor),labels = data
		if is_cuda:
			query_tensor,passage_tensor,labels=query_tensor.cuda(),passage_tensor.cuda(),labels.cuda()
		# zero the parameter gradients
		optimizer.zero_grad()

		# forward + backward + optimize
		outputs = model(query_tensor,passage_tensor)
		loss = criterion(outputs, labels.view(-1))
		loss.backward()
		optimizer.step()
		# print statistics
		running_loss += loss.item()
		if i%100==0:
			print(f"Running Loss {running_loss}  Loss {loss.item()} epoch {epoch} i {i}")
			running_loss = 0.0

def valid(model,valid_dl,criterion,epoch,is_cuda=True):
	running_loss=0
	accuracy=0
	correct=0
	model.eval()
	for i, data in enumerate(valid_dl, 0):
		# get the inputs
		(query_tensor,passage_tensor),labels = data
		if is_cuda:
			query_tensor,passage_tensor,labels=query_tensor.cuda(),passage_tensor.cuda(),labels.cuda()
		# forward + backward + optimize
		outputs = model(query_tensor,passage_tensor)
		loss = criterion(outputs, labels.view(-1))
		max_index = outputs.max(dim = 1)[1]
		correct+=(max_index == labels.view(-1)).sum()
		running_loss += loss.item()
		if i%100==0:
			print(f"Running Loss {running_loss} Loss {loss.item()} epoch {epoch} i {i} Correct {correct}")
			running_loss = 0.0
	print(f"Accuracy {correct/(i*8)}")

def main():
	parser = argparse.ArgumentParser()
	dataset=MRCDataset("../data/data_h5py/")
	indices=range(1,dataset.__len__())
	train_indices=indices[1:int(0.9*len(indices))]
	valid_indices=indices[-int(0.9*len(indices)):]
	train_sampler=SubsetRandomSampler(train_indices)
	valid_sampler=SubsetRandomSampler(valid_indices)
	batch_size=8
	train_dl = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler,num_workers=4)
	valid_dl = DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler,num_workers=4)

	model=SimpleCNNModel().cuda()

	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(model.parameters(), lr=5e-4)
	is_cuda=True
	NUM_OF_EPOCHS=5
	# for epoch in range(NUM_OF_EPOCHS):  # loop over the dataset multiple times
	# 	train(model,optimizer,train_dl,criterion,epoch)
	# 	if epoch%1==0:
	# 		valid(model,valid_dl,criterion,epoch)
	# torch.save(model,open("simplemodel.pth","wb"))
	databunch=DataBunch(train_dl,valid_dl,device=torch.device('cuda'))
	learner=Learner(databunch,model,loss_func=criterion,metrics=accuracy,callback_fns=ShowGraph)
	learner.lr_find()
	learner.fit_one_cycle(1,5e-5)
	learner.save("fastaimodel.pth")
	print('Finished Training')



if __name__ == '__main__':
	main()