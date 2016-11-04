import csv
import re
import os
import sys
import numpy as np
import time
import collections
import argparse

# transform row data to day form


def readfile(name):
	f = open(name,'r')
	weeks = np.empty(shape=[0,337])
	empty_week = np.zeros(337).astype(float)
	#weeks = np.append(weeks,[empty_week],axis=0)
	#ddd =f.read().splitlines()
	record = -1 #record if it is a new week
	recId = -1 # record if it is a new house

	for rows in f.read().splitlines():
		row = rows.split(',')
		#print(data)
		day = (float(row[1])//100-191)%7; # [0]Sun [1]Mon [2]Tue ..
		if recId != row[0]:
			weeks = np.append(weeks,[empty_week],axis=0)
			weeks[-1,0] = row[0]
			recId = row [0]
		if day == 0 and record !=day:
			weeks = np.append(weeks,[empty_week],axis=0)
			weeks[-1,0] = row[0]
       
		time = (float(row[1])-1)%48; #[1] 12:00 [2] 12:15
		pos = int(day*48+time+1);
		weeks[-1,pos]=float(row[2])
		record = day
        
	return weeks

def chooseWeek (weeks, number):

	id_list = np.unique(weeks[:,0])
	#print(id_list)
	ref = np.zeros([len(id_list),337])

	for i in range(len(id_list)):
		indx = np.where(weeks[:,0]==id_list[i])
		pos = indx[0][number-1]
		ref[i,:] = weeks[pos,:]
	return ref, id_list

def getlabel(label,house_list):
	
	f = open('survey_.dat','r')
	idx = len(house_list)
	y = np.zeros([idx,1])
	choose = -1

	if label =='ageperson':
		choose = 0
	elif label =='allemployed':
		choose = 1
	elif label == 'cooking':
		choose = 2
	elif label =='employment':
		choose = 3
	elif label == 'family':
		choose =4 
	elif label =='income':
		choose = 5
	elif label == 'Children':
		choose = 6
	elif label == 'retirement':
		choose = 7
	elif label == 'single':
		choose = 8
	elif label == 'unoccupied':
		choose = 9

	dic = {}
	if choose == -1:
		print ('Please select one of labels')
	else:
		for rows in f.read().splitlines():
			row = rows.split(',')
			dic[float(row[0])] = row[choose*2+1]

	#print (dic)
	for i in range(idx):
		if house_list[i] in dic.keys():
			y[i,0] = dic[house_list[i]]
		else:
			y[i,0] = -1
	# choose week 
	# get id
	#pick label
	#print (y)
	return y
def nonlin(x,deriv=False):
	if(deriv==True):
	    return x*(1-x)

	return 1/(1+np.exp(-x))

def trainNN (trainX,trainY):
#NN    
	X = trainX [:,1:-1]
	y = trainY
	idx = X.shape

	np.random.seed(1)

	# randomly initialize our weights with mean 0
	syn0 = 2*np.random.random([idx[1],idx[0]]) - 1
	syn1 = 2*np.random.random((idx[0],1)) - 1

	for j in xrange(60000):

		# Feed forward through layers 0, 1, and 2
		l0 = X
		l1 = nonlin(np.dot(l0,syn0))
		l2 = nonlin(np.dot(l1,syn1))

		# how much did we miss the target value?
		l2_error = y - l2
    
		if (j% 10000) == 0:
			print "Error:" + str(np.mean(np.abs(l2_error)))
        
		# in what direction is the target value?
		# were we really sure? if so, don't change too much.
		l2_delta = l2_error*nonlin(l2,deriv=True)

		# how much did each l1 value contribute to the l2 error (according to the weights)?
		l1_error = l2_delta.dot(syn1.T)
    
		# in what direction is the target l1?
		# were we really sure? if so, don't change too much.
		l1_delta = l1_error * nonlin(l1,deriv=True)

		syn1 += l1.T.dot(l2_delta)
		syn0 += l0.T.dot(l1_delta)

np.set_printoptions(suppress=True)
r = readfile('small.txt')
[tt,t2] = chooseWeek(r,1)
temp = getlabel('ageperson',t2)
trainNN(tt,temp)