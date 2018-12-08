"""
Simple Model:
Input:
Sentence Embedding:[Q,P]

Output:
Relevance:[0,1]*10
"""
import torch
import torch.nn.functional as F
import torch.nn as nn 

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
		
class QuerySentCompare(nn.Module):
	"""docstring for QuerySentCompare"""
	def __init__(self, name):
		super(QuerySentCompare, self).__init__()
		self.name=self.name
		self.convbranch = nn.Sequential(F.relu(nn.Conv2d(6,16,kernel_size=3)),
										nn.MaxPool2d(2),
										F.relu(nn.Conv2d(16,32,kernel_size=3)),
										nn.MaxPool2d(2),
										Flatten()
										)
	def forward(self,query,passage):
		inp=torch.cat([query,passage],dim=0)
		inp=inp.view(-1,6,16,16)
		return self.convbranch(inp)

class Model(nn.Module):
    def __init__(self,num_of_passages=10):
        super(Model, self).__init__()
        self.num_of_passages=num_of_passages
        self.query_sent_compare_branches=[QuerySentCompare(f"convb_{i}") for i in range(self.num_of_passages)]
        self.convbranch = nn.Sequential(F.relu(nn.Conv2d(20,16,kernel_size=3)),
										nn.MaxPool2d(2),
										Flatten()
										)
        self.dense=nn.Sequential(nn.Linear(144,64),
        						F.sigmoid(nn.Linear(64,10)))

    def forward(self, query,passages):
    	passages=list(torch.split(passages,1,dim=0))
    	inp=torch.cat([self.query_sent_compare_branches[i](passages[i])for i in range(self.num_of_passages)],dim=0)
    	inp=inp.view(-1,20,8,8)
    	inp=self.convbranch(inp)
    	inp=dense(inp)
    	return inp



