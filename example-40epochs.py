#!/usr/bin/env python
# coding: utf-8

# Results for the first 40 epochs



import random
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, train_test_split
import torch
import time




device = 'cuda' if torch.cuda.is_available() == True else 'cpu'


import numpy as np

def new_data2(file,lines):
  d = {}
  ind2w = {0: "SOS", 1: "EOS"}
  lets = 2
  f  = open(file)
  mass = []
  mws = [] # molec weigths
  forms = [] # molec form
  types_ = [] # types nan, MS1, MS2 : 0, 1, 2
  mspec = []
  smi = []
  c = 0
  l = 0
  for line in f:
    l +=1
    if len(line.split()) != 2:
        c = 1
    else:
        c = 0
    if c == 1:
      smi.append(line.split(',')[0])
      mws.append(float(line.split(',')[-1]))
 
      forms.append(np.array([int(ii) for ii in line.split(',')[4:10]]))
    #  forms.append([line.split(',')[4:10]])
      if line.split(',')[1] == 'nan':
        types_.append(0)
      else:
        if line.split(',')[1] == 'MS1':
          types_.append(1)
        elif line.split(',')[1] == 'MS2':
          types_.append(2)

      for letter in line.split(',')[0]:
        if letter not in d:
          d[letter] = int(lets)
          ind2w[lets] = letter
          lets += 1
    if c == 0:
        mspec.append((float(line.split()[0]),float(line.split()[1])))
    if c == 1 or l == lines:
      mass.append(mspec)
      mspec = []

  return np.array(mass), np.array(mws), np.array(forms),np.array(types_), np.array(smi), d, ind2w




f = 'path_to_file'
num_lines = sum(1 for line in open(f))




x,x2,x3,x4,y,dic,ind2w = new_data2(f,num_lines)






x = np.delete(x,0)




print(len(x),len(x2),len(x3),len(y))




#REMOVE OUTLIER; MOLECULES WITH MASS WEIGHT > 100
maxw=300

for wei in reversed(x2):
  if wei > maxw:
    idx = np.where(x2 == wei)

  #
    y = np.delete(y,idx)
    x = np.delete(x,idx)
    x2 = np.delete(x2,idx)
    x3 = np.delete(x3,idx,axis=0)
    x4 = np.delete(x4,idx)




print(len(y),len(x),len(x2),len(x3),len(x4))




import matplotlib.pyplot as plt
lens={}
ml = 0
for smi in y:
  if len(smi) > ml:
    ml = len(smi)
  else:
    ml = ml
  if len(smi) not in lens:
    lens[len(smi)] = 1
  else:
    lens[len(smi)] += 1
plt.bar(lens.keys(),lens.values())
print('max len', ml)




## CHECK RANDOM
print(x[1333],x2[1333],x3[1333],x4[1333],y[1333])




X_train1, X_val1,X_train2,X_val2,X_train3,X_val3,X_train4,X_val4,y_train,y_val = train_test_split(x,x2,x3,x4,y,test_size=0.1,random_state=42,shuffle=True) # 





print(len(X_train1),len(X_val1))




import math
import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

# Encoder as Set transformer classes based from juho lee https://arxiv.org/pdf/1810.00825.pdf

class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K):
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        A = torch.softmax(Q_.bmm(K_.transpose(1,2))/math.sqrt(self.dim_V), 2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O

class SAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, ln=False):
        super(SAB, self).__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(X, X)

class ISAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
        super(ISAB, self).__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)

    def forward(self, X):
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X)
        return self.mab1(X, H)

class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(self.S.repeat(X.size(0), 1, 1), X)



class SetTransformer(nn.Module):
    def __init__(self, dim_input, num_outputs, dim_output,
            num_inds, dim_hidden, num_heads, ln=False):
        super(SetTransformer, self).__init__()
        self.enc = nn.Sequential(
                ISAB(dim_input, dim_hidden, num_heads, num_inds, ln=ln),
                ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln))
        self.dec = nn.Sequential(
                PMA(dim_hidden, num_heads, num_outputs, ln=ln),
                SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
                SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
                nn.Linear(dim_hidden, dim_output))

    def forward(self, X):
        return self.dec(self.enc(X))


#Decoder part


class DecoderRNN(nn.Module):
        def __init__(self, embed_size, hidden_size, vocab_size, num_layers, dp, max_length=ml):
                super(DecoderRNN, self).__init__()
                self.embed = nn.Embedding(vocab_size, embed_size)
                #self.lstm = nn.LSTM(embed_size, hidden_size*2, num_layers, batch_first=True, dropout=dp)
                self.lstm = nn.LSTM(embed_size, hidden_size+8, num_layers, batch_first=True, dropout=dp)

             #   self.out = nn.Linear(hidden_size*2, vocab_size)
                self.out = nn.Linear(hidden_size+8, vocab_size)

                self.softmax = nn.LogSoftmax(dim=1)

        def forward(self, in_smiles ,hidden,cell):#
                embeddings = self.embed(in_smiles).view(1,1,-1)
                embeddings = F.relu(embeddings)
                output,(hidden, cell) = self.lstm(embeddings,(hidden,cell))
                outputs = self.softmax(self.out(output[0]))
                return outputs, hidden, cell




tf = 0.75 # teacher forcing rate
VOC = len(ind2w)




def tensor_from_smiles(smiles_b):
        indexes = [dic[let] for let in smiles_b]
        indexes.append(int(1)) # 0 : EOS token
        in_smiles = torch.LongTensor(indexes).to(device)
        

        return in_smiles




def get_layers(N,HID_DIM,hidden_st):
  
  # for creating inputs to the LSTM with n layer:
  temp = torch.randn(N,1,HID_DIM).to(device)
  for lay in range(N):
    temp[lay] = hidden_st
  hidden1 = temp
  return hidden1




def evaluate(encoder1, decoder,x,x2,x3,x4,smi,LAY,HID):

	loss2 = 0
	hidden0 = encoder1(x)

	hidden2 = torch.cat((hidden0.squeeze(0),x2,x3,x4),dim=1)
	hidden2 = get_layers(LAY,HID+8,hidden2)

	cell = hidden2
	hidden = hidden2
	pretarget = tensor_from_smiles(smi)
	target = torch.LongTensor([[0]]).to(device)

	for di in range(len(pretarget)):
		output, hidden, cell = decoder(target,hidden,cell)
		topv,topi = output.data.topk(1)
		target = topi.squeeze().unsqueeze(dim=0).detach()
		loss2 += criterion(output,pretarget[di].unsqueeze(dim=0) )
		if int(topi[0][0]) == 1: break
        

	voss = loss2.item()/len(pretarget)
	return  voss

def evaval(encoder1,decoder,test_pairs,LAY,HID):
	voss=0
	for i in range(len(test_pairs[0])):

		X = torch.tensor(test_pairs[0][i]).unsqueeze(dim=0).to(device)
		X2 = torch.tensor(test_pairs[1][i]).unsqueeze(dim=0).unsqueeze(dim=0).float().to(device)
		X3 = torch.tensor(np.array(test_pairs[2][i])).unsqueeze(dim=0).float().to(device)
		X4 = torch.tensor(test_pairs[3][i]).unsqueeze(dim=0).unsqueeze(dim=0).float().to(device)

		smi = test_pairs[4][i]
		voss += evaluate(encoder1,decoder,X,X2,X3,X4,smi,LAY,HID)
	voss_prom = voss/(len(test_pairs[0]))
	return voss_prom


def evaluateR(encoder1,decoder,x,x2,x3,x4,LAY,HID,max_length=ml): # puede cambiarse a numero muy grande
        hidden0 = encoder1(x)

        hidden2 = torch.cat((hidden0.squeeze(0),x2,x3,x4),dim=1)
        hidden2 = get_layers(LAY,HID+8,hidden2)      

        hidden = hidden2
        cell = hidden2
        target =torch.LongTensor([[0]]).to(device)#0: SOS
        decoded_words = []
        for di in range(max_length):
                output, hidden, cell = decoder(target,hidden,cell)
                topv,topi = output.data.topk(1)
                if int(topi[0][0]) == int(1): # EOS 
                        decoded_words.append('<EOS>')
                        break
                else:
                        decoded_words.append(ind2w[int(topi[0][0])])
                target = topi.squeeze().unsqueeze(dim=0).detach() ### detach?
        return decoded_words

def evaluateRandomly(encoder1,decoder,ppair,LAY,HID,n=50):
        for i in range(n):
                choice = random.randint(0,len(ppair)-1)
                xt = torch.tensor(ppair[0][choice]).unsqueeze(dim=0).to(device)
                x2t = torch.tensor(ppair[1][choice]).unsqueeze(dim=0).unsqueeze(dim=0).float().to(device)
                x3t = torch.tensor(np.array(ppair[2][choice])).unsqueeze(dim=0).float().to(device)
                x4t = torch.tensor(ppair[3][choice]).unsqueeze(dim=0).unsqueeze(dim=0).float().to(device)

                smi = ppair[4][choice]
		
                output_words = evaluateR(encoder1,decoder,xt,x2t,x3t,x4t,LAY,HID)
                output_s = ''.join(output_words)
                print("pred:",output_s)
                print("real:",smi)
def evaluateTodo(encoder1,decoder,ppair,LAY,HID):
        for i in range(len(ppair[0])):
                X = torch.tensor(ppair[0][i]).unsqueeze(dim=0).to(device)
                X2 = torch.tensor(ppair[1][i]).unsqueeze(dim=0).unsqueeze(dim=0).float().to(device)
                X3 = torch.tensor(np.array(ppair[2][i])).unsqueeze(dim=0).float().to(device)
                X4 = torch.tensor(ppair[3][i]).unsqueeze(dim=0).unsqueeze(dim=0).float().to(device)

                smi = ppair[4][i]
                output_words = evaluateR(encoder1,decoder,X,X2,X3,X4,LAY,HID)
                output_s = ''.join(output_words)
                print("pred:",output_s)
                print("real:",smi)




criterion = nn.NLLLoss()




def model_function():

	EMB = 256
	HID= 256
	LAY= 2
	enc_UNITS = 32
	INDS = 32
	HEADS = 4
	DP_dec =	0.1
	l_r = 1e-4
	

	decoder = DecoderRNN(EMB,HID,VOC,LAY,DP_dec).to(device)
	encoder1 = SetTransformer(2,1,HID,INDS,enc_UNITS,HEADS).to(device)

	
	epochs = 40
	encoder_optimizer1 = optim.AdamW(encoder1.parameters(),lr = l_r)

	decoder_optimizer = optim.AdamW(decoder.parameters(),lr = l_r) ##

	x_tpair = [X_train1,X_train2,X_train3,X_train4,y_train]
	test_pairs = [X_val1,X_val2,X_val3,X_val4,y_val]
	b_size = len(X_train1)
	px = [] # plot train loss
	py = []
	ppx = [] # plot val loss
	ppy = []
	p = 0 # for print every 500 iterations (see below)
	for epoch in range(1,epochs+1):
		start_time = time.time()
		encoder1.train()


		decoder.train()
		sum_loss = 0


		for b in range(b_size):  
			p +=1

			decoder_optimizer.zero_grad()
			encoder_optimizer1.zero_grad()


			loss = 0
			
			smis = str(y_train[b])
			X_train11 = torch.tensor(X_train1[b]).unsqueeze(dim=0).to(device)
			X_train22 = torch.tensor(X_train2[b]).unsqueeze(dim=0).unsqueeze(dim=0).float().to(device)
			X_train33 = torch.tensor(np.array(X_train3[b])).unsqueeze(dim=0).float().to(device)
			X_train44 = torch.tensor(X_train4[b]).unsqueeze(dim=0).unsqueeze(dim=0).float().to(device)

			pretarget = tensor_from_smiles(smis)
	 
			hidden0=encoder1(X_train11) # inithidden , features from encoder output

			hidden2 = torch.cat((hidden0.squeeze(0),X_train22,X_train33,X_train44),dim=1)
			hidden2 = get_layers(LAY,HID+8,hidden2)

			hidden = hidden2
			cell = hidden2
			target = torch.LongTensor([0]).to(device)#0: SOS
			use_tf = True if random.random() < tf else False
			if use_tf:
				for s in range(len(pretarget)):
					output,hidden,cell  = decoder(target,hidden,cell)
					target = pretarget[s].unsqueeze(dim=0)

					loss += criterion(output, target)
			else:
				for s in range(len(pretarget)):
					output,hidden,cell  = decoder(target,hidden,cell)
					topv,topi = output.data.topk(1)
					target = topi.squeeze().unsqueeze(dim=0).detach()
					loss += criterion(output, pretarget[s].unsqueeze(dim=0))
		
			loss.backward()
			sum_loss += loss.item()/len(pretarget)
	 		

			encoder_optimizer1.step()
			decoder_optimizer.step() 
	 
			if p % 500 == 0:
				print('train loss: ', loss.item()/len(pretarget),'iteration: ',p)
		
		px.append(epoch)
	
		py.append(sum_loss/b_size)
		print('epoch: ',epoch, 'train loss: ',sum_loss/b_size)	    

		encoder1.eval()

		decoder.eval()

		voss = evaval(encoder1,decoder,test_pairs,LAY,HID)
		print('epoch: ',epoch, 'test loss: ',voss)	    
		ppx.append(epoch)
		ppy.append(voss)
		duration = time.time() - start_time
		print('duration: ',duration)	
		plt.plot(px,py,label='Train Loss')
		plt.plot(ppx,ppy,label='Validation Loss')
		plt.legend()
		plt.xlabel('Epochs')
		plt.ylabel('Loss')
		plt.show()
		torch.save(encoder1.state_dict(),'path_to_savefile'+str(epoch)+'_w')
		torch.save(decoder.state_dict(),'path_to_savefile'+str(epoch)+'_w')
		torch.save(encoder_optimizer1.state_dict(),'path_to_savefile'+str(epoch)+'_w')
		torch.save(decoder_optimizer.state_dict(),'path_to_savefile'+str(epoch)+'_w')
		if epoch%5 == 0 :  # evaluate smiles every N epochs and save model

			print('train eval: ')
			evaluateRandomly(encoder1,decoder,x_tpair,LAY, HID)

			print('test eval: ')
			evaluateTodo(encoder1,decoder,test_pairs,LAY,HID)

model_function()

