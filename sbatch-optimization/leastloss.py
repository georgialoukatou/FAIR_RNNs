import os 

import re



list=[]

filename = '/scratch2/gloukatou/fair_project/char-lm-code-master/sbatch-optimization/Yucatec1_randomopt.txt'
with open(filename) as f:
	lines = [line.rstrip('\n') for line in f]
	for line in lines:
		result = re.search('minimum loss=(.*) epoch', line)
		if result:
			loss = re.findall('\d+\.?\d+', result.group(0))
			list.append(loss)
        #print(list)
	#print(min(list))


mix=1000
for num in list:
    if float(num[0]) < mix:
     mix = float(num[0])

print(mix)	
			
