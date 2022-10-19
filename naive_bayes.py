'''
GPL3.0 License new update 19102022
'''

import numpy as np
from joblib import dump, load 

def separate_data(text):
	new = []
	data_pos = []
	data_neg = []
	for x in range(len(text)):
		for y in range(len(text[x].split())):
			if text[x].split()[y] == 'y':
				new = text[x].split()
				new.remove('y')
				data_pos.append(new)
			elif text[x].split()[y] == 'n':
				new = text[x].split()
				new.remove('n')
				data_neg.append(new)
			else:
				continue
	return data_pos, data_neg

def remove_non_char(text): #remove symbolic character from data
	new_text = []
	for x in range(len(text)):
		rep = []
		for y in range(len(text[x])):
			for z in range(len(text[x][y])):
				if text[x][y][z].isalpha():
					replaced = text[x][y]
				else:
					replaced = str(text[x][y]).replace(text[x][y][z], '')
			rep.append(replaced.lower())							
		new_text.append(rep)
	return new_text

def get_unique_word(text_pos, text_neg):
	unique = []
	a = []
	a.extend(text_pos)
	a.extend(text_neg)
	for x in range(len(a)):
		for y in range(len(a[x])):
			if len(unique) > 0:
				for z in range(len(unique)):
					if a[x][y] == unique[z]:
						break
					else:
						if z == (len(unique)-1):
							unique.append(a[x][y])		
						else:
							continue
			else:
				unique.append(a[x][y])
	return unique

def cal_prob(unique, tpos, tneg):
	num_a = len(tpos) + len(tneg)
	ocp = np.array(occurance(unique, tpos))
	ocm = np.array(occurance(unique, tneg))  
	for x in range(len(unique)):
		for y in range(len(ocp)):
			if ocp[y] == unique[x]:
				np.put(ocp, y+1, ((int(ocp[y+1])+1)/(int(ocp[len(ocp)-1]) + int(len(unique)))))
				np.put(ocm, y+1, ((int(ocm[y+1])+1)/(int(ocm[len(ocm)-1]) + int(len(unique)))))
				break
			else:
				continue
	np.put(ocp, int(len(ocp)-1), len(tpos)/num_a)
	np.put(ocm, int(len(ocm)-1), len(tneg)/num_a)
	return ocp, ocm

def occurance(unique, text):
	a = []
	c = 0
	for x in range(len(unique)):
		b = 0
		for y in range(len(text)):
			for z in range(len(text[y])):
				if unique[x] == text[y][z]:
					b += 1
				else:
					continue
		c = c + b		
		a.append(unique[x])	
		a.append(str(b))
	a.append(str(c))
	return a

def test_prob(text, model):
	r = 1
	for x in range(len(text)):
		for y in range(len(model)):
			if text[x] == model[y]:
				r = r * float(model[y+1])
				break
			else:
				continue
	return r*float(model[len(model)-1])

if __name__ == "__main__":

	print("[INFO] Train probability... ")
	data_file = open('data_file.txt', 'r')
	data_in = data_file.readlines()
	data_pos, data_neg = separate_data(data_in)
	data_pos = remove_non_char(data_pos)
	data_neg = remove_non_char(data_neg)
	words = get_unique_word(data_pos, data_neg)
	prob_pos, prob_neg = cal_prob(words, data_pos, data_neg)

	dump(prob_pos, "prob_pos.pb")
	dump(prob_neg, "prob_neg.pb")

	print("-".ljust(50, '-'))
	print("probability of each words".ljust(50,'-'))
	print("positive label: ")
	print(prob_pos)
	print("negative label: ")
	print(prob_neg)
	print("-".ljust(50, '-'))
	
	print("[INFO] your model has been saved...")

	print("[INFO] Test our model probability...")
	print("[INFO] Input your text...")
	test_text = input()
	pos = test_prob(test_text.lower().split(), prob_pos)
	neg = test_prob(test_text.lower().split(), prob_neg)

	print("-".ljust(50, '-'))
	print("result of the test".ljust(50,'-'))
	print("positive probability {}". format(pos/(pos+neg)))
	print("negative probability {}". format(neg/(pos+neg)))
	print("")
	print("Your input data: {}". format(test_text))
	if pos > neg:
		print("it is positive")
	else:
		print("it is negative")
	print("-".ljust(50, '-'))
	exit()
