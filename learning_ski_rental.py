# Program : To verify the regret of the of the ski-rental problem in multiple stages using the hedge algorithm. At each stage the predictions are the buy cost.
# Date : 21-4-2020
# Author : Anant Shah
# E-Mail : anantshah200@gmail.com

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import random

def get_adv_loss(b_pred,x_pred,b_t,x_t) :
	# Only 2 strategies, either buy at day 0 or rent forever
	w = []
	for i in range(num_experts) :
		if b_pred[i] < x_pred[i] :
			# Expert is advising to buy at t=0
			cost = b_t
		else :
			# Strategy is to rent for all times
			cost = x_t
		w.append(cost)
	return w

# We assume that one of the experts is true

T = 1000
num_experts = 5
B_max = 200
B_min = 1
eps = 0.1
#mu = 0.0
#sigma = 2

b_true = np.random.randint(low=B_min,high=B_max,size=(T,1)) # the true buy values at each instant
x_adv = np.random.randint(low=1,high=2*B_max,size=(T,1))
print(b_true.shape)

w_t = np.ones((num_experts,1)) # The initial weights assigned to each expert
print(w_i.shape)

true_exp = np.random.randint(low=0,high=num_experts,size=1)
print(true_exp)

loss = 0.0

for t in range(T) :
	b_pred = np.random.randint(low=B_min,high=B_max,size=(num_experts,1))
	#eps = np.random.normal(mu,sigma,num_experts)
	#b_pred = b_true[t] + eps
	x_pred = np.random.randint(low=1,high=2*B_max,size=(num_experts,1))
	b_pred[true_exp] = b_true[t]
	x_pred[true_exp] = x_adv[t]
	p_t = w_t / np.sum(w_t)
	# Calculate the cost vector <m>
	m = get_adv_loss()
	w_t = w_t * np.exp(-eps*m)
	loss += np.dot(p_t,m)
	

