# Program : Replicate the results from 1.) Kumar et al 
# Author : Anant Shah
# Date : 26-3-2019
# E-mail : anantshah200@gmail.com

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import random

##################### Robust Algorithm #####################

num_trials = 10000
mu = 0.0
num_std = 500
sigmas = np.linspace(0,250,num_std)
#eps = np.random.normal(mu,sigma,num_trials) # Noise to be added to the actual prediction
b = 100 # Cost of buying the skis
b_samples = np.random.uniform(1.0,4*b,num_trials)

lam = 0.5
comp_ratios_det = []

for sigma in range(num_std) :

	comp_ratio = 0
	eps = np.random.normal(mu,sigmas[sigma],num_trials)

	for trial in range(num_trials) :
		x = b_samples[trial] # True number of ski-ing days
		y_pred = x + eps[trial]

		if y_pred >= b :
			buy_day = np.ceil(lam*b)
			if x >= buy_day :
				alg = b + buy_day - 1
			else :
				alg = x
		elif y_pred < b :
			buy_day = np.ceil(b/lam)
			if x >= buy_day :
				alg = b + buy_day - 1
			else :
				alg = x

		if x >= b :
			opt = b
		else :
			opt = x

		comp_ratio += alg / opt

	comp_avg = comp_ratio / num_trials
	comp_ratios_det.append(comp_avg)

#plt.plot(sigmas,comp_ratios)
#plt.grid('True')
#plt.xlabel('Standard deviation')
#plt.ylabel('Competitive Ratio')
#plt.show()

#################################### Randomized Algorithm #########################

comp_ratios_ran = []

lam = np.log(1.5)

for sigma in range(num_std) :

	comp_ratio = 0
	eps = np.random.normal(mu,sigmas[sigma],num_trials)

	for trial in range(num_trials) :
		x = b_samples[trial] # True number of ski-ing days
		y_pred = x + eps[trial]

		if y_pred >= b :
			k = np.floor(lam*b)
			ind = np.arange(1,k+1)
			p_x = np.power((b-1)/b,k-ind)*(1.0/(b*(1-np.power((1-1/b),k))))
			P_x = np.cumsum(p_x)
			rand_num = random.uniform(0,1)
			buy_day = 1
			for i in range(int(k)-1) :
				if rand_num>=P_x[i] and rand_num<P_x[i+1] :
					buy_day = i + 2
					break
			if x >= buy_day :
				alg = b + buy_day - 1
			else :
				alg = x
		elif y_pred < b :
			l = np.ceil(b/lam)
			ind = np.arange(1,l+1)
			p_x = np.power((b-1)/b,l-ind)*(1.0/(b*(1-np.power((1-1/b),l))))
			P_x = np.cumsum(p_x)
			rand_num = random.uniform(0,1)
			buy_day = 1
			for i in range(int(l)-1) :
				if rand_num>=P_x[i] and rand_num<P_x[i+1] :
					buy_day = i + 2
					break
			if x >= buy_day :
				alg = b + buy_day - 1
			else :
				alg = x

		if x >= b :
			opt = b
		else :
			opt = x

		comp_ratio += alg / opt
	
	comp_avg = comp_ratio / num_trials
	comp_ratios_ran.append(comp_avg)

plt.plot(sigmas,comp_ratios_det)
plt.plot(sigmas,comp_ratios_ran)
plt.grid('True')
plt.legend([r'Deterministic $\lambda = 0.5$',r'Randomized $\lambda = ln(1.5)$'])
plt.xlabel('Standard deviation')
plt.ylabel('Competiitve Ratio')
plt.show()
