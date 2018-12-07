"""Main File to """
import os
import argparse
import random
import logging
from tqdm import tqdm,trange

import numpy as np 
import pandas as pd 

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertForSequenceClassification
from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s', 
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)



def main():
	parser = argparse.ArgumentParser()

	##Required parameters
	parser.add_argument("--data_dir",
						default="../data/",
						type=str,
						help="The input data dir.Should contain tsv")
	parser.add_argument("--model",
						default="simple",
						type="str",
						help="Type of model to select to train ")


if __name__ == '__main__':
	main()