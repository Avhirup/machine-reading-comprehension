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



def main():
	parser = argparse.ArgumentParser()

	##Required parameters
	# parser.add_argument("--data_dir",
	# 					default="../data/",
	# 					type="str",
	# 					help="The input data dir.Should contain tsv")
	# parser.add_argument("--model",
	# 					default="simple",
	# 					type="str",
	# 					help="Type of model to select to train ")

	dataset=MRCDataset("../data/data_rows/")
	indices=range(dataset.__len__())
	train_indices=indices[0:int(0.9*len(indices))]
	valid_indices=indices[-int(0.9*len(indices)):]
	train_sampler=SubsetRandomSampler(train_indices)
	valid_sampler=SubsetRandomSampler(valid_indices)
	batch_size=2
	train_dl = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
	valid_dl = DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler)

	model=SimpleCNNModel().cuda()
	# databunch=DataBunch(train_dl,valid_dl)
	# learner=Learner(databunch,SimpleCNNModel())
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(model.parameters(), lr=1e-3)
	is_cuda=True

	for epoch in range(1):  # loop over the dataset multiple times

	    running_loss = 0.0
	    for i, data in enumerate(train_dl, 0):
	        # get the inputs
	        query_tensor,passage_tensor,labels = data
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
	        if i % 20 == 1999:    # print every 2000 mini-batches
	            print('[%d, %5d] loss: %.3f' %
	                  (epoch + 1, i + 1, running_loss / 2000))
	            running_loss = 0.0

	

	print('Finished Training')



if __name__ == '__main__':
	main()