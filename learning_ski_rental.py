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
	# First obtain the optimal cost we will suffer
	if b_t < x_t :
		opt = b_t
	else :
		opt = x_t
	for i in range(num_experts) :
		if b_pred[i] < x_pred[i] :
			# Expert is advising to buy at t=0
			cost = b_t
		else :
			# Strategy is to rent for all times
			cost = x_t
		if cost == opt :	
			w.append(0)
		else :
			w.append(1)
	return np.array(w)

# We assume that one of the experts is true

T = 10000
num_experts = 1000
B_max = 1
B_min = 0
eps = np.sqrt(np.log(num_experts)/T)

b_true = np.random.uniform(low=B_min,high=B_max,size=T) # the true buy values at each instant
x_adv = np.random.uniform(low=0,high=2*B_max,size=T)

w_t = np.ones(num_experts) # The initial weights assigned to each expert

true_exp = np.random.randint(low=0,high=num_experts,size=1)

loss = 0.0
regret = []
cumul_m = np.zeros(num_experts) # Keep a track on the mistakes made by each expert

for t in range(T) :
	b_pred = np.random.uniform(low=B_min,high=B_max,size=num_experts)
	#eps = np.random.normal(mu,sigma,num_experts)
	#b_pred = b_true[t] + eps
	x_pred = np.random.uniform(low=0,high=2*B_max,size=num_experts)
	#b_pred[true_exp] = b_true[t]
	#x_pred[true_exp] = x_adv[t]
	p_t = w_t / np.sum(w_t)
	# Calculate the cost vector <m>
	m = get_adv_loss(b_pred,x_pred,b_true[t],x_adv[t])
	w_t = w_t * np.exp(-eps*m)
	loss += np.dot(p_t,m)
	cumul_m = cumul_m + m
	min_loss = np.amin(cumul_m)
	# We have a true expert so the loss we are comparing with is basically 0
	regret.append(loss-min_loss)

plt.plot(range(1,T+1),regret)
plt.xlabel('Time')
plt.ylabel('Regret')
plt.grid('True')
plt.show()
