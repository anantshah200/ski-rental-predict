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

T = 5000
B_max = 200
B_min = 100
#for sigma in sigmas :
#	b_true = np.random.uniform(low=B_min,high=B_max,size=T)
#	x_adv = np.random.uniform(low=B_min,high=2*B_max,size=T)
#	loss = 0.0
#	true_exp = np.random.randint(low=0,high=num_experts,size=1)
#	w_t = np.ones(num_experts)
#	regret = []
#	cumul_m = np.zeros(num_experts) # Keep a track on the mistakes by each expert
#	for t in range(T) :
#		b_pred = np.random.uniform(low=B_min,high=B_max,size=num_experts)
#		eps_b = np.random.normal(mu,sigma,num_experts)
#		b_pred = b_pred + eps_b
#		eps_x = np.random.normal(mu,sigma,num_experts)
#		#b_pred = b_true[t] + eps
#		x_pred = np.random.uniform(low=B_min,high=2*B_max,size=num_experts)
#		x_pred = x_pred + eps_x
#		#b_pred[true_exp] = b_true[t]
#		#x_pred[true_exp] = x_adv[t]
#		p_t = w_t / np.sum(w_t)
#		# Calculate the cost vector <m>
#		m = get_adv_loss(b_pred,x_pred,b_true[t],x_adv[t])
#		w_t = w_t * np.exp(-eps*m)
#		loss += np.dot(p_t,m)
#		cumul_m = cumul_m + m
#		min_loss = np.amin(cumul_m)
#		# We have a true expert so the loss we are comparing with is basically 0
#		regret.append(loss-min_loss)
#	sigma_regret.append(regret)
#	
#for i in range(len(sigmas)) :
#	plt.plot(range(1,T+1),sigma_regret[i])
#plt.legend([r'$\sigma=0.5$',r'$\sigma=1.0$',r'$\sigma=1.5$',r'$\sigma=2.0$'])
#plt.xlabel('Time')
#plt.ylabel('Regret')
#plt.grid('True')
#plt.show()

###################### Want to test if the advice is strategies(predicted x and a certain lambda) #######################

# sigmas already declared before

#def get_adv_strat_loss(lam_pred,xx_pred,b_t,x_t) :
#	# lambda corresponds to the paper by Purohit et al

#for sigma in sigmas :
#	b_true = np.random.uniform(low=B_min,high=B_max,size=T)
#	x_adv = np.random.uniform(low=B_min,high=2*B_max,size=T)
#	w_t = np.ones(num_experts)
#	regret = []
#	cumul_m = np.zeros(num_experts)
#	for t in range(T) :
#		lam_pred = np.random.uniform(low=0.0,high=1.0,size=num_experts) # The strategy predictions by each expert
#		eps_b = np.random.normal(mu,sigma,num_experts)
#		x_pred = np.random.uniform(low=B_min,high=2*B_max,size=num_experts)
#		x_pred = x_pred + eps_b # Number of skiing days prediction by each expert
#		b_t = b_true[t] # The buy price for the current timestep
#		x_t = x_adv[t] # The number of skiing days for the current timestep(not aware to the experts)
#		p_t = w_t / np.sum(w_t)
#		# get adversary loss for each expert
#		m = get_adv_strat_loss(lam_pred,x_pred,b_t,x_t)
#		w_t = w_t * np.exp(-eps*m)
#		loss += np.dot(p_t,m)
#		cumul_m = cumul_m + m
#		min_loss = np.amin(cumul_m)
#		regret.append(loss-min_loss)

#################################################

# Check the dependence of the regret on the lambda parameters. In the paper by Purohit et al, the deterministic and ranodmized algorithm have  a consistency and robustness tradeoff as a function of the parameter lambda. Now we want to see in the online learning setup what is the dependence of the regret on this parameter lambda. At each time-step we have a set of predictors, predicting the number of ski-ing days and some predicting the buy cost. Split them into bins with each bin having a certain standard deviation. Now sample a certain buy cost(b') and based on the predictions run the algorithm by Purohit et al. Compare this loss to the true algorithm whether to rent for all or buy at day 0 and check the regret. Compare for different values of lambda, check the effect of number of experts. Check the effect of standard deviation etc. We need to get the predictions for the number of ski-ing days, not randomly sample as there is no relation in the current implementation. Take an average of the prediction based on the weights and then give that to all the experts predicting the ski-days.

# Experiments : 1.] Want to see how bad this implementation is versus had we told the experts the true buy cost : result surprisingly is not that bad !
# 2.] Varied parameters lambda from 0.4 to 0.6. At 0.6 the regret was essentially the same. 
# 3.] Now want to see if I vary the buy-cost experts what is the effect

def get_pred_loss_rand(b_t,x_t,b_sample,x_pred,lam,num_experts) :
	"Function to obtain the loss based on the randomized algorithm"
	# Arguments : b_t : The true buy cost
	#	      x_t : The true numbe of ski-ing days
	#	      b_sample : The value of buy-cost sampled. Not equal to the true buy cost
	#	      x_pred : The predictions for the number of ski-days by the experts
	#	      lam : Tradeoff parameter

	# First obtain the optimal value that could be suffered by the agent
	if b_t >= x_t :
		opt = x_t # rent for all days
	elif b_t < x_t :
		opt = b_t # buy at day 0
	
	m = [] # Mistake vector that stores the ratio for each expert

	# Now cost incurred if we follow each expert(here we follow the randomized algorithm by Purohit et al.)
	for expert in range(num_experts) :
		if x_pred[expert] >= b_sample :
			k = np.floor(lam*b_sample)
			ind = np.arange(1,k+1)
			p_x = np.power((b_sample-1)/b_sample,k-ind)*(1.0/(b_sample*(1-np.power((1-1/b_sample),k))))
			P_x = np.cumsum(p_x)
			rand_num = random.uniform(0,1)
			buy_day = 1
			for i in range(int(k)-1) :
				if rand_num>=P_x[i] and rand_num<P_x[i+1] :
					buy_day = i + 2
					break
			if x_t>=buy_day :
				alg = b_t + buy_day - 1 # When we are actually buying we incur the true cost
			else :
				alg = x_t
		elif x_pred[expert] < b_sample :
			l = np.ceil(b_sample/lam)
			ind = np.arange(1,l+1)
			p_x = np.power((b_sample-1)/b_sample,l-ind)*(1.0/(b_sample*(1-np.power((1-1/b_sample),l))))
			P_x = np.cumsum(p_x)
			rand_num = random.uniform(0,1)
			buy_day = 1
			for i in range(int(l)-1) :
				if rand_num>=P_x[i] and rand_num<P_x[i+1] :
					buy_day = i + 2
					break
			if x_t >= buy_day :
				alg = b_t + buy_day - 1
			else : 
				alg = x_t
		loss = alg/opt # The competitive ratio
		m.append(loss)
	m = np.array(m) - 1 # Since we want to match with the best ratio which is 1
	assert np.all(m>=0)
	return np.array(m) # Note that all losses will be greater than 1. The best case scenario will be when the ratio is 1

def get_pred_loss_det(b_t,x_t,b_sample,x_pred,lam,num_experts) :
	"Function to get the loss for the deterministic algorithm"
	# Arguments :   b_t : The true buy cost at the current time instant
	#		x_t : The true number of ski-ing days at the current instant
	#		b_sample : The buy cost sampled for the current instant(not the true buy cost)
	#		x_pred : The vector of predictions for the number of ski-ing days
	#		lam : Tradeoff parameter

	# Optimal cost suffered by the agent
	if b_t >= x_t :
		opt = x_t # Rent for all days
	else :
		opt = b_t # Buy at day 0

	m = [] # The loss for each expert

	# Cost of following each expert
	for expert in range(num_experts) :
		if x_pred[expert] >= b_sample :
			buy_day = np.ceil(lam*b_sample)
			if x_t >= buy_day :
				alg = b_t + buy_day - 1
			else :
				alg = x_t
		elif x_pred[expert] < b_sample :
			buy_day = np.ceil(b_sample/lam)
			if x_t >= buy_day :
				alg = b_t + buy_day - 1
			else :
				alg = x_t
		loss = alg / opt
		m.append(loss)

	m = np.array(m) - 1
	assert np.all(m>=0)
	return m

lams = [0.2,0.4,0.6,0.8] # Based on the algorithm by Purohit et al
lam = 0.4
b_true = np.random.randint(low=B_min,high=B_max,size=T)
x_adv = np.random.randint(low=1,high=4*B_max,size=T)
num_bins = 5 # Categorize the experts into bins based on their error of prediction
#bin_vals = [2,4,8,20]
#bin_ind = []
#rand_bins = np.random.permutation(num_experts)
#sigmas = [1,20] # The number of sigmas will be equal to the number of bins
#num_buy = 50

#for i in range(num_bins) :
#	bin_ind.append(rand_bins[int(i*num_experts/num_bins):int((i+1)*num_experts/num_bins)])

lam_regret = []
#expert_range = [100] # Experts for predicting the number of ski-ing days
num_experts = 100
eps = np.sqrt(np.log(num_experts)/T)
buy_experts = 500 # The experts predicting the buy cost
eta_b = np.sqrt(np.log(buy_experts)/T)
buy_num_bins = 5 # The numbers of bins the buy experts will be binned into
b_sigmas = np.linspace(1,20,buy_num_bins) # The standard deviaiot of error for the bins for the experts predicting buy costs
buy_rand_bins = np.random.permutation(buy_experts) # How we divide these bins
buy_bin_ind = [] # To store the bin indices

for i in range(buy_num_bins) :
	buy_bin_ind.append(buy_rand_bins[int(i*buy_experts/buy_num_bins):int((i+1)*buy_experts/buy_num_bins)])

loss_fault = []
true_opt = []

rand_bins = np.random.permutation(num_experts)
Eps_x = np.zeros((T,num_experts)) # Noise for the experts predicting ski days
Eps_b = np.zeros((T,buy_experts)) # Noise for the experts predicting buy cost
bin_ind = []
for i in range(num_bins) :
	bin_ind.append(rand_bins[int(i*num_experts/num_bins):int((i+1)*num_experts/num_bins)])
sigmas = np.linspace(1,20,num_bins)

for i in range(num_bins) :
	Eps_x[:,bin_ind[i]] = np.random.normal(0.0,sigmas[i],(T,int(num_experts/num_bins)))

for i in range(buy_num_bins) :
	Eps_b[:,buy_bin_ind[i]] = np.random.normal(0.0,b_sigmas[i],(T,int(buy_experts/buy_num_bins)))

for j in range(2) :

	# 1 iteration for the faulty buy costs
	# 1 iteration for the case when the true buy cost is revealed
	#rand_bins = np.random.permutation(num_experts)
	#eps_x = np.zeros(num_experts)
	#eps_b = np.zeros(buy_experts)
	#bin_ind = []
	#for i in range(num_bins) :
	#	bin_ind.append(rand_bins[int(i*num_experts/num_bins):int((i+1)*num_experts/num_bins)]) 
	loss = 0.0
	#sigmas = np.linspace(1,20,num_bins)
	regret = []
	cumul_m = np.zeros(num_experts)
	w_t = np.ones(num_experts) # weights for the experts predicting the number of ski-ing days
	b_w_t = np.ones(buy_experts) # weights for experts predicting the buy cost of the ski-is

	for t in range(T) :

		#for i in range(num_bins) :
		#	eps_x[bin_ind[i]] = np.random.normal(0.0,sigmas[i],int(num_experts/num_bins))
		#for i in range(buy_num_bins) :
		#	eps_b[buy_bin_ind[i]] = np.random.normal(0.0,b_sigmas[i],int(buy_experts/buy_num_bins))
		eps_x = Eps_x[t]
		eps_b = Eps_b[t]
		b_pred = b_true[t] + eps_b # The predictions of the buy cost
		if j == 0 :
			#b_sample = 2*b_true[t]
			b_p_t = b_w_t / np.sum(b_w_t)
			b_sample = np.dot(b_p_t,b_pred) # The weighted average will give us a predicted buy cost
		# Sample the b' prediction
		x_pred = x_adv[t] + eps_x # The predicitons by the experts for the current time-instant

		# now use algorithm by Purohit et al. to obsreve the loss and compare to the optimal
		b_t = b_true[t] # The true buy cost at the current time instant
		x_t = x_adv[t] # The true number of ski-ing days. Not known to the algorithm. Used to calculate optimal cost
		p_t = w_t/np.sum(w_t)

		if j == 1 :
			# True buy cost revealed to the experts(just repeating Purohit et al for each iteration)
			b_sample = b_t

		m = get_pred_loss_rand(b_t,x_t,b_sample,x_pred,lam,num_experts) # Vector of competitive ratios

		loss += np.dot(p_t,m)

		# Note that a higher ratio implies a larger loss so we should the decrease the weight of that expert more
		w_t = w_t * np.exp(-eps*m)
		b_w_t = b_w_t * np.exp(-eta_b*np.absolute((b_pred-b_t)/b_t))

		# Calculate the regret for this case
		if j == 0 :

			m = get_pred_loss_rand(b_t,x_t,b_t,x_pred,lam,num_experts) # Want to compare with if true price told
			cumul_m = cumul_m + m
			min_loss = np.amin(cumul_m)
			regret.append(loss-min_loss)
		else :

			cumul_m = cumul_m + m
			min_loss = np.amin(cumul_m)
			regret.append(loss-min_loss)

	lam_regret.append(regret)

# Expect to be in descending order
for i in range(num_bins) :
	print(np.sum(p_t[bin_ind[i]]))

p_b_t = b_w_t/np.sum(b_w_t) # Probability distribution over the buy experts
for i in range(buy_num_bins) :
	print(np.sum(p_b_t[buy_bin_ind[i]]))

for i in range(len(lam_regret)) :
	plt.plot(range(1,T+1),lam_regret[i])
#plt.legend([r'$\lambda = 0.2$',r'$\lambda = 0.4$',r'$\lambda = 0.6$',r'$\lambda = 0.8$'])
plt.legend([r'Faulty Buy Experts',r'True Buy Cost'])
plt.xlabel('Time')
plt.ylabel('Regret')
plt.title(r'Comparison of faulty experts vs true expert $\lambda = %1.2f$'%lam)
plt.grid('True')
plt.show()
