
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
		self.name=name
		self.convbranch = nn.Sequential(nn.Conv2d(6,16,kernel_size=3),
										nn.BatchNorm2d(16),
										nn.ReLU(),
										nn.MaxPool2d(2),
										nn.Conv2d(16,32,kernel_size=3),
										nn.BatchNorm2d(32),
										nn.MaxPool2d(2),
										Flatten()
										)
	def forward(self,query,passage):
		inp=torch.cat([query,passage],dim=0)
		inp=inp.view(-1,6,16,16)
		return self.convbranch(inp).unsqueeze(1)

class Model(nn.Module):
    def __init__(self,num_of_passages=10):
        super(Model, self).__init__()
        self.num_of_passages=num_of_passages
        # self.query_sent_compare_branches=[QuerySentCompare(f"convb_{i}") for i in range(self.num_of_passages)]
        self.query_sent_compare_branch0=QuerySentCompare("convb_0")
        self.query_sent_compare_branch1=QuerySentCompare("convb_1")
        self.query_sent_compare_branch2=QuerySentCompare("convb_2")
        self.query_sent_compare_branch3=QuerySentCompare("convb_3")
        self.query_sent_compare_branch4=QuerySentCompare("convb_4")
        self.query_sent_compare_branch5=QuerySentCompare("convb_5")
        self.query_sent_compare_branch6=QuerySentCompare("convb_6")
        self.query_sent_compare_branch7=QuerySentCompare("convb_7")
        self.query_sent_compare_branch8=QuerySentCompare("convb_8")
        self.query_sent_compare_branch9=QuerySentCompare("convb_9")
        self.convbranch = nn.Sequential(nn.Conv2d(20,16,kernel_size=3),
        								nn.BatchNorm2d(16),
        								nn.ReLU(),
										nn.MaxPool2d(2),
										Flatten()
										)
        self.dense=nn.Sequential(nn.Linear(144,64),
        						nn.Linear(64,10),
        						)

    def forward(self, query,passages):
    	passages=list(torch.split(passages,1,dim=1))
    	inp0=self.query_sent_compare_branch0(query,passages[0])
    	inp1=self.query_sent_compare_branch1(query,passages[1])
    	inp2=self.query_sent_compare_branch2(query,passages[2])
    	inp3=self.query_sent_compare_branch3(query,passages[3])
    	inp4=self.query_sent_compare_branch4(query,passages[4])
    	inp5=self.query_sent_compare_branch5(query,passages[5])
    	inp6=self.query_sent_compare_branch6(query,passages[6])
    	inp7=self.query_sent_compare_branch7(query,passages[7])
    	inp8=self.query_sent_compare_branch8(query,passages[8])
    	inp9=self.query_sent_compare_branch9(query,passages[9])
    	inp=torch.cat([inp0,inp1,inp2,inp3,inp4,inp5,inp6,inp7,inp8,inp9],dim=1)
    	inp=inp.view(-1,20,8,8)
    	inp=self.convbranch(inp)
    	inp=self.dense(inp)
    	return inp



