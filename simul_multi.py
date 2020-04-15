# Program : Implement Gollapudi et al.. Ski-Rental problem with multiple predictors
# Author : Anant Shah
# Date : 1-4-2020
# E-Mail : anantshah200@gmail.com

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import random
from scipy.special import comb
from scipy.stats import halfnorm
from scipy.stats import truncnorm
import scipy.stats as stats

########################## Ski-Rental Multiple Experts : True Expert ######################

num_trials = 10000
b = 1 # Buying cost. Can be scaled
num_std = 5
sigmas = np.linspace(0.0,2.0,num_std)
mu = 0.0
num_experts = 8
upper = 3*b # Upper value for the truncated distribution

x_samples = np.random.uniform(0.0,2*b,num_trials) # Obtain samples for the number of skiing days
x_samples = x_samples.reshape((x_samples.shape[0],1))
comp = []
for expert in range(1,num_experts+1) :
	comp_ratios = [] # List to store the competitive ratios for each expert set
	# Calculate break points for every number of experts
	# Obtain them from the system of equations derived in the paper
	poly = np.ones(expert+1)
	poly[expert] = -1
	bp_1 = [x for x in np.roots(poly) if x.imag==0]
	bp_1 = bp_1[-1].real

	bp_list = np.power(bp_1,np.arange(1,expert))
	bp_list = np.cumsum(bp_list)*b
	bp_list = np.append(bp_list,b)
	bp_list = np.insert(bp_list,0,0.0)

	print(bp_list) # Print the breakpoints

	for sigma in sigmas :

		comp_sum = 0.0 # To obtain the average competitive ratio
		for i in range(num_trials) :

			x = x_samples[i]
			#eps = np.random.normal(0.0,sigma,size=expert)
			#eps = halfnorm.rvs(loc=0,scale=sigma/halfnorm.std(loc=0,scale=1),size=expert)
			X = truncnorm((-x-0.0)/sigma,(upper-0.0)/sigma,loc=0.0,scale=sigma) # The noise ranges from -x to an upper bound so that the predictions still remain positive
			eps = X.rvs(size=expert)
			#print(halfnorm.std(loc=-x,scale=sigma/halfnorm.std(loc=0,scale=1)))
			#eps = np.random.normal(0,sigma,expert)
			#true_exp = np.random.randint(0,expert,size=1)
			#eps[true_exp] = 0.0
			#min_err = np.amin(np.absolute(eps))
			#print(min_err)
			y = x + eps
			exit = 0

			for j in range(1,expert+1) :
		
				init_val = bp_list[j-1] # The initial value of the range
				fin_val = bp_list[j] # Final value of the range
				# If we do not find a prediction in the range, rent till the start of the range and buy at that point

				if not np.any(((y>=init_val) & (y<fin_val))) :
					exit = 1
					if x > init_val :
						alg = b + init_val
						if x >= b :
							opt = b
						else :
							opt = x
					else : 
						# since we are renting till the start of the interval
						alg = x
						opt = x
					break
			if exit == 0 :
				# All the ranges have a prediction
				alg = x
				if x >=b :
					opt = b
				else :
					opt = x
	
			comp_sum += alg / opt

		comp_avg = comp_sum / num_trials
		comp_ratios.append(comp_avg)
	comp.append(comp_ratios)

for i in range(num_experts) :
	plt.plot(sigmas,comp[i])
plt.grid('True')
plt.legend(['k=1','k=2','k=3','k=4','k=5','k=6','k=7','k=8'])
plt.xlabel('Standard Deviation')
plt.ylabel('Competitive Ratio')
plt.title('Consistent Algorithm')
plt.show()

####################### Robust Algorithm ##################################

num_experts = 6


#Obtain the breakpoints based on the system of equations in Algorithm 2

poly_coeff = []
poly_coeff.append(1.0/np.power(2,num_experts-1))
if num_experts >=3 :
	poly_coeff.append((np.cumsum(np.power(-1.0,np.arange(1,num_experts-1)))[-1]+1)/np.power(2,num_experts-2)-np.power(-1,num_experts-1)/np.power(2,num_experts-1))
else :
	poly_coeff.append(1/np.power(2,num_experts-2)-np.power(-1,num_experts-1)/np.power(2,num_experts-1))
for i in np.arange(num_experts-2,0,-1) :
	combs = comb(num_experts-1-i+np.arange(0,i),np.arange(0,i))
	mult = np.power(-1,np.arange(0,i))
	temp_sum = np.dot(combs,mult) / np.power(2,i-1) - np.power(-1,i)*comb(num_experts-1,i,exact=True) / np.power(2,i)
	poly_coeff.append(temp_sum)
poly_coeff.append(-1)

bp_z = [x for x in np.roots(poly_coeff) if x.imag==0]
bp_z = bp_z[-1].real

assert bp_z>=0 and bp_z<=1, "Not correct first breakpoint"

bps_z = []
bps_z.append(bp_z)
for i in range(1,num_experts-1) :
	bp_next = bp_z*(1+bps_z[i-1]/2)/(1-bp_z/2)
	bps_z.append(bp_next)
bps_z.append(1.0)
bps_z = np.array(bps_z)
bps_z = np.insert(bps_z,0,0.0)
comp_ratios = []

for sigma in sigmas :
	
	comp_sum = 0.0
	experts = np.arange(1,num_experts+1)

	for i in range(num_trials) :
		
		x = x_samples[i]
		N = truncnorm((-x-0.0)/sigma,(upper-0.0)/sigma,loc=0.0,scale=sigma)
		eps = N.rvs(num_experts)
		#min_err = np.amin(np.absolute(eps))
		#eps = np.random.normal(0.0,sigma,num_experts)
		#min_err = np.amin(np.absolute(eps))
		y = x + eps
		exit = 0

		for j in range(1,num_experts) :
			
			init_val = bps_z[j-1]*b
			fin_val = bps_z[j]*b

			if not np.any(((y>=init_val)&(y<fin_val))) :
				exit = 1
				if j == 1:
					alg = b
				else :
					if x >= (init_val + fin_val)/2 :
						alg = b + (init_val+fin_val)/2
					else :
						alg = x
				break
		if x >= b :
			opt = b
		else :
			opt = x
		if exit == 0 :
			alg = x
		comp_sum += alg / opt
	
	comp_avg = comp_sum / num_trials
	comp_ratios.append(comp_avg)

plt.plot(sigmas,comp_ratios)
plt.grid('True')
plt.title('Robust Algorithm')
plt.xlabel('Standard Deviation')
plt.ylabel('Competitive Ratio')
plt.show()

###################### Hybrid Algorithm (Robust and Consistent) ##################

# Get the appropriate breakpoints for the hybrid case

#lam = 0.9 # Parameter to obtain a good competitive ratio
num_experts = 8
lams = [0.1,0.5,0.9] # Algorithm will not buy in the [0,b*lam] range

#poly_coeff = np.power(1.0/(1+lam),np.arange(num_experts,0,-1))
#poly_coeff[0] = poly_coeff[0] * (1+lam)
#poly_coeff = np.append(poly_coeff,-1)
#bp_hyb = [x for x in np.roots(poly_coeff) if x.imag==0]
#bp_hyb = bp_hyb[-1].real

#assert bp_hyb<=1 and bp_hyb>=0,"Not valid breakpoint"

# Obtain rest of the breakpoints
#bps_hyb = []
#bps_hyb.append(bp_hyb)

#for i in range(1,num_experts-1) :
#	cur_bp = bps_hyb[0]*(1+bps_hyb[i-1])/(1+lam)
#	bps_hyb.append(cur_bp)

#bps_hyb = np.array(bps_hyb)
#bps_hyb = np.insert(bps_hyb,0,lam)
#bps_hyb = np.append(bps_hyb,1.0)

#assert bps_hyb.shape[0] == (num_experts+1), "Size does not match"

comp_lam = []
for lam in lams :
	poly_coeff = np.power(1.0/(1+lam),np.arange(num_experts,0,-1))
	poly_coeff[0] = poly_coeff[0] * (1+lam)
	poly_coeff = np.append(poly_coeff,-1)
	bp_hyb = [x for x in np.roots(poly_coeff) if x.imag==0]
	bp_hyb = bp_hyb[-1].real

	assert bp_hyb<=1 and bp_hyb>=0,"Not valid breakpoint"

	# Obtain rest of the breakpoints
	bps_hyb = []
	bps_hyb.append(bp_hyb)

	for i in range(1,num_experts-1) :
		cur_bp = bps_hyb[0]*(1+bps_hyb[i-1])/(1+lam)
		bps_hyb.append(cur_bp)

	bps_hyb = np.array(bps_hyb)
	bps_hyb = np.insert(bps_hyb,0,lam)
	bps_hyb = np.append(bps_hyb,1.0)
	comp_ratios = []
	
	for sigma in sigmas :
	
		comp_sum = 0.0
	
		experts = np.arange(1,num_experts+1)
		eps = np.random.normal(mu,sigma,num_experts*num_trials)
		eps = eps.reshape((num_trials,num_experts))

		y_pred = x_samples + eps # Need to somehow manage the negative predictions

		for i in range(num_trials) :

			x = x_samples[i]
			y = y_pred[i]
			exit = 0

			for j in range(1,num_experts+1) :
			
				init_val = bps_hyb[j-1]*b
				fin_val = bps_hyb[j]*b

				if not np.any(((y>=init_val)&(y<fin_val))) :
					exit = 1
					if x > init_val :
						alg = b + init_val
					else :
						alg = x
					break
		
			if x >= b :
				opt = b
			else :
				opt = x

			if exit == 0:
				alg = x
			comp_sum += alg / opt
	
		comp_avg = comp_sum / num_trials
		comp_ratios.append(comp_avg)
	comp_lam.append(comp_ratios)

for i in range(len(lams)) :
	plt.plot(sigmas,comp_lam[i])
plt.legend([r'$\lambda = 0.1$',r'$\lambda = 0.5$',r'$\lambda = 0.9$'])
plt.grid('True')
plt.xlabel('Standard deviation')
plt.ylabel('Competitive Ratio')
plt.title('Hybrid Algorithm')
plt.show()
