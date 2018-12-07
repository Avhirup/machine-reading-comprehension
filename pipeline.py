import pandas as pd 
import numpy as np 









def main():
	data_path="~/workspace/MSAIC/data/"
	train_data_path=data_path+"traindata.tsv"
	eval_data_path=data_path+"validationdata.tsv"

	batch_size=32
	traindataiter=pd.read_csv(train_data_path,sep="\t",chunksize=batch_size,header=None,names=['QID','Query','Passage','IsRelevant','PassageId'])
	traindata=traindataiter.get_chunk()



if __name__ == '__main__':
	main()