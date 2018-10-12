#! /bin/env python

from config import CHECKPOINT_HOME

#PATH='/fair_project/char-lm-code-master/'


#infile=open('/fair_project/char-lm-code-master/results.txt', 'r') as infile:
path_to_infile=(CHECKPOINT_HOME + '/results.txt')


loss=[]

def process(path_to_infile):
	def minimal_loss():
		with open(path_to_infile, 'r') as infile:
		#def minimal_loss():		
			for line in infile:
				line_=line.split()
				if len(loss)<1:
					loss.append(line_[-1])
				else:
					if all(i > line_[-1] for i in loss):
						loss[0]=line_[-1]
						return(loss[0])
	minimal_loss()
	def corresp():
		with open(path_to_infile, 'r') as infile:	
			for line in infile:
				if loss[0] in line:
					return(line)
	return corresp()
	

print(process(path_to_infile))
