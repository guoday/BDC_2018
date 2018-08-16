def mykey(x):
	return -x[1]
A=[]
A_prob=[]
with open('output/xgb_dev.txt','r') as f:
	for line in f:
		line=line.strip().split(',')
		A.append(line[0])
		A_prob.append((line[0],float(line[1])))
A_prob.sort(key=mykey)
print(len(A))

B=[]
B_prob=[]
with open('output/lgb_dev.txt','r') as f:
	for line in f:
		line=line.strip().split(',')
		B.append(line[0])
		B_prob.append((line[0],float(line[1])))
B_prob.sort(key=mykey)
print(len(B))



A=set(A)
B=set(B)

Interaction=A&B
A=[]
B=[]




for key,_ in A_prob:
	if key not in Interaction:
		A.append([key,_])
for key,_ in B_prob:
	if key not in Interaction:
		B.append([key,_])
	
print(len(Interaction),len(A),len(B))


dev_label=[]
with open('output/dev_label.txt','r') as f:
	for line in f:
		dev_label.append(line.strip())
best=0
rate=None
dev_label=set(dev_label)
for i in range(100):
	for j in range(100):
			rate_A=0.01*i
			rate_B=0.01*j
			left=set()
			for key,prob in A:
				if prob>=rate_A:
					left.add(key)			
			for key,prob in B:
				if prob>=rate_B:
					left.add(key)
			res=Interaction|left
			recall=len(res&dev_label)*1.0/len(res)
			precision=len(res&dev_label)*1.0/len(dev_label)
			f1=recall*precision*2/(recall+precision)
			if f1>best:
				print(f1,len(res))
				best=f1
				rate=(rate_A,rate_B)
print(best)
print(rate)

A=[]
A_prob=[]
with open('output/xgb_test.txt','r') as f:
	for line in f:
		line=line.strip().split(',')
		A.append(line[0])
		A_prob.append((line[0],float(line[1])))
A_prob.sort(key=mykey)
print(len(A))

B=[]
B_prob=[]
with open('output/lgb_test.txt','r') as f:
	for line in f:
		line=line.strip().split(',')
		B.append(line[0])
		B_prob.append((line[0],float(line[1])))
B_prob.sort(key=mykey)
print(len(B))



A=set(A)
B=set(B)
Interaction=A&B
A=[]
B=[]




for key,_ in A_prob:
	if key not in Interaction:
		A.append([key,_])
for key,_ in B_prob:
	if key not in Interaction:
		B.append([key,_])
	
print(len(Interaction),len(A),len(B))
for key,prob in A:
	if prob>=rate[0]:
		left.add(key)			
for key,prob in B:
	if prob>=rate[1]:
		left.add(key)

res=Interaction|left

with open('result.txt','w') as f:
	for r in res:
		f.write(str(r)+'\n')

