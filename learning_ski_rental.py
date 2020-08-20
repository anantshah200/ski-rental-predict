# Program : To verify the regret of the of the ski-rental problem in multiple stages using the hedge algorithm. At each stage the predictions are the buy cost.
# Date : 21-4-2020
# Author : Anant Shah
# E-Mail : anantshah200@gmail.com

#import mpl_toolkits
#from mpl_toolkits.mplot3d import Axes3D

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import random

# Check the dependence of the regret on the lambda parameters. In the paper by Purohit et al, the deterministic and ranodmized algorithm have  a consistency and robustness tradeoff as a function of the parameter lambda. Now we want to see in the online learning setup what is the dependence of the regret on this parameter lambda. At each time-step we have a set of predictors, predicting the number of ski-ing days and some predicting the buy cost. Split them into bins with each bin having a certain standard deviation. Now sample a certain buy cost(b') and based on the predictions run the algorithm by Purohit et al. Compare this loss to the true algorithm whether to rent for all or buy at day 0 and check the regret. Compare for different values of lambda, check the effect of number of experts. Check the effect of standard deviation etc. We need to get the predictions for the number of ski-ing days, not randomly sample as there is no relation in the current implementation. Take an average of the prediction based on the weights and then give that to all the experts predicting the ski-days.

# Experiments : 1.] Want to see how bad this implementation is versus had we told the experts the true buy cost : result surprisingly is not that bad ! Obs : If i increase the number of buy experts, the algorithm performs worse than the optimal. Now lets change the number of experts predicting the number of ski-ing days.
# 2.] Varied parameters lambda from 0.4 to 0.6. At 0.6 the regret was essentially the same. 
# 3.] Now want to see if I vary the buy-cost experts what is the effect

def get_pred_loss_rand(b_t,x_t,b_s,x_pred,lam,num_experts) :
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
	
	m = np.zeros(num_experts) # Mistake vector that stores the ratio for each expert

	# Now cost incurred if we follow each expert(here we follow the randomized algorithm by Purohit et al.)
	for expert in range(num_experts) :
		if x_pred[expert] >= b_s :
			k = np.floor(lam*b_s)
			ind = np.arange(1,k+1)
			b_s = int(np.ceil(b_s))
			p_x = np.power((b_s-1)/b_s,k-ind)*(1.0/(b_s*(1-np.power((1-1/b_s),k))))
		
			#print(p_x)
			#print(np.sum(p_x))
			#p_x[-1] += 1.0 - np.sum(p_x)
			#print(np.sum(p_x))
			#P_x = np.cumsum(p_x)
			#print(P_x[-1])
			#print(P_x[-1])
			#rand_num = random.uniform(0,1)
			#buy_day = 1
			#for i in range(int(k)-1) :
			#	if rand_num>=P_x[i] and rand_num<P_x[i+1] :
			#		buy_day = i + 2
			#		break
			#print(k)
			buy_day = np.random.choice(int(k),1,p=p_x) + 1
			if x_t>=buy_day :
				alg = b_t + buy_day - 1 # When we are actually buying we incur the true cost
			else :
				alg = x_t
		elif x_pred[expert] < b_s :
			l = np.ceil(b_s/lam)
			ind = np.arange(1,l+1)
			b_s = int(np.ceil(b_s))
			p_x = np.power((b_s-1)/b_s,l-ind)*(1.0/(b_s*(1-np.power((1-1/b_s),l))))
			p_x[-1] += 1.0 - np.sum(p_x)
			#print(np.sum(p_x))
			#P_x = np.cumsum(p_x)
			#print(P_x[-1])
			
			#rand_num = random.uniform(0,1)
			#buy_day = 1
			#for i in range(int(l)-1) :
			#	if rand_num>=P_x[i] and rand_num<P_x[i+1] :
			#		buy_day = i + 2
			#		break
			buy_day = np.random.choice(int(l),1,p=p_x) + 1
			if x_t >= buy_day :
				alg = b_t + buy_day - 1
			else : 
				alg = x_t
		loss = alg/opt # The competitive ratio
		m[expert] = loss
	m = m - 1 # Since we want to match with the best ratio which is 1
	assert np.all(m>=0)
	#print(np.amax(m))
	#assert np.all(m<=1)
	return m # Note that all losses will be greater than 1. The best case scenario will be when the ratio is 1

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

def get_action(dist,size) :
	# function to obtain an auction based on the distribution
	# Arguments :   dist - The distribution
	# 		size - The size of the distribution

	ran = np.random.uniform(0.0,1)
	arm = 0

	dist = np.cumsum(dist)

	if ran < dist[0] :
		return 0
	for i in range(size-1) :
		if ran >= dist[i] and ran < dist[i+1] :
			arm = i + 1
			break
	return arm

def get_b_samples(b_p_t,b_pred,buy_experts,num_experts) :
	# function to obtaint the sampled buy cost for each ski expert
	# Arguments :	b_p_t : The current probability distribution over buy experts
	#		b_pred : The prediction made by eaach buy expert

	rand_vals = np.random.uniform(0.0,1.0,num_experts)
	b_sample = np.zeros(num_experts)

	dist = np.cumsum(b_p_t)

	for i in range(num_experts) :
		ran = rand_vals[i]
		if ran < dist[0] :
			b_sample[i] = b_pred[0]
			continue
		for j in range(buy_experts-1) :
			if ran >= dist[j] and ran < dist[j+1] :
				b_sample[i] = b_pred[j+1]
				break
	return b_sample


# I need to iterate over different ranges of supports for the ski days and the ski cost

T = 10000

B_min = 250
B_max = 350

#lams = [5,10,50]
lams = [15]
num_bins = 5 # Categorize the experts into bins based on their error of prediction

#lam_regret = []
b_regret = []
per_agent_regret = []
overlap = [] # The overlapping area between the supports as a decimal
#num_experts = 100
#eps = np.sqrt(np.log(num_experts)/T)
#buy_experts = 15 # The experts predicting the buy cost
#eta_b = np.sqrt(np.log(buy_experts)/T)

buy_num_bins = 5 # The numbers of bins the buy experts will be binned into

b_sigmas = np.linspace(1,40,buy_num_bins) # The standard deviaiot of error for the bins for the experts predicting buy costs
#buy_rand_bins = np.random.permutation(buy_experts) # How we divide these bins
#buy_bin_ind = [] # To store the bin indices

#for i in range(buy_num_bins) :
#	buy_bin_ind.append(buy_rand_bins[int(i*buy_experts/buy_num_bins):int((i+1)*buy_experts/buy_num_bins)])

#rand_bins = np.random.permutation(num_experts)
#Eps_x = np.zeros((T,num_experts)) # Noise for the experts predicting ski days
#Eps_b = np.zeros((T,buy_experts)) # Noise for the experts predicting buy cost
#bin_ind = []

#for i in range(num_bins) :
#	bin_ind.append(rand_bins[int(i*num_experts/num_bins):int((i+1)*num_experts/num_bins)])
sigmas = np.linspace(1,40,num_bins)

#for i in range(num_bins) :
#	Eps_x[:,bin_ind[i]] = np.random.normal(0.0,sigmas[i],(T,int(num_experts/num_bins)))

#for i in range(buy_num_bins) :
#	Eps_b[:,buy_bin_ind[i]] = np.random.normal(0.0,b_sigmas[i],(T,int(buy_experts/buy_num_bins)))

#for lam in lams :
num_experts = 15

# Standard deviation range
#sig_range = 19

var_sims = 15
#lam_regret = np.zeros((var_sims,T))

avg_lam_regret = []

lam = 0.5

for buy_experts in lams :

	lam_regret = np.zeros((var_sims,T))

	#b_sigmas = np.linspace(init,init+sig_range,num_bins)

	for sims in range(var_sims) :

		print(sims)

		num_bins = 5
		buy_num_bins = 5

		b_true  = np.random.randint(low=B_min,high=B_max,size=T)
		x_adv = np.random.randint(low=B_min,high=B_max,size=T)
		#b_sigmas = np.linspace(start_sig,start_sig+sig_range,buy_num_bins)
		#print(sigmas)

		eps = np.sqrt(np.log(num_experts)/T)
		#eta_b = np.sqrt(np.log(buy_experts)/T)

		rand_bins = np.random.permutation(num_experts)
		eps_x = np.zeros(num_experts)
		eps_b = np.zeros(buy_experts)
		Eps_x = np.zeros((T,num_experts))
		Eps_b = np.zeros((T,buy_experts))
		bin_ind = []
		if num_bins > num_experts :
			num_bins = num_experts
		for i in range(num_bins) :
			bin_ind.append(rand_bins[int(i*num_experts/num_bins):int((i+1)*num_experts/num_bins)]) 

		eta_b = np.sqrt(np.log(buy_experts)/T)

		buy_rand_bins = np.random.permutation(buy_experts) # How we divide the buy experts
		buy_bin_ind = []

		if buy_num_bins > buy_experts :
			buy_num_bins = buy_experts
		for i in range(buy_num_bins) :
			buy_bin_ind.append(buy_rand_bins[int(i*buy_experts/buy_num_bins):int((i+1)*buy_experts/buy_num_bins)])
		
		Eps_b = np.zeros((T,buy_experts))

		for i in range(buy_num_bins) :
			Eps_b[:,buy_bin_ind[i]] = np.random.normal(0.0,b_sigmas[i],(T,int(buy_experts/buy_num_bins)))

		for i in range(num_bins) :
			Eps_x[:,bin_ind[i]] = np.random.normal(0.0,sigmas[i],(T,int(num_experts/num_bins)))

		#print(lam)

		loss = 0.0
		#sigmas = np.linspace(1,20,num_bins)
		regret = []
		b_diff = []
		agent_regret = [] # List to store the regret per agent
		cumul_m = np.zeros(num_experts)
		cumul_ski_loss = np.zeros(num_experts) # To store the cumulative loss for the ski experts
		w_t = np.ones(num_experts) # weights for the experts predicting the number of ski-ing days
		b_w_t = np.ones(buy_experts) # weights for experts predicting the buy cost of the ski-is

		add = 0

		reg_loss1 = []
		reg_loss2 = []

		ski_cumul_loss = 0.0

		#true_buy_expert =np.random.randint(buy_experts,size=1) # True environment predictor case
		true_buy_expert = 0

		Eps_b[:,0] = 0.0

		for t in range(T) :

			eps_x = Eps_x[t]
			eps_b = Eps_b[t]

			#eps_b[true_buy_expert] = 0.0

			b_pred = b_true[t] + eps_b # The predictions of the buy cost
			b_p_t = b_w_t / np.sum(b_w_t)
			#b_sample = np.dot(b_p_t,b_pred) # The weighted average will give us a predicted buy cost
			#b_sample = get_b_samples(b_p_t,b_pred,buy_experts,num_experts) # Get the sampled buy cost for each expert

			sampled_buyer = np.random.choice(buy_experts,1,p=b_p_t)

			# Sample the b' prediction
			x_pred = x_adv[t] + eps_x # The predicitons by the experts for the current time-instant

			# now use algorithm by Purohit et al. to obsreve the loss and compare to the optimal
			b_t = b_true[t] # The true buy cost at the current time instant
			x_t = x_adv[t] # The true number of ski-ing days. Not known to the algorithm. Used to calculate optimal cost
			p_t = w_t/np.sum(w_t)

			#x_exp = get_action(p_t,num_experts)
			ski_exp_loss = np.zeros((buy_experts,num_experts))
			buy_upd = np.zeros(buy_experts)

			for buyer in range(buy_experts) :
				m = get_pred_loss_rand(b_t,x_t,b_pred[buyer],x_pred,lam,num_experts) # Vector of competitive ratios
				#print((m*b_p_t[buyer]).shape)
				ski_exp_loss[buyer] =  m*b_p_t[buyer]
				buy_upd[buyer] = np.sum(m*p_t)
				#print(ski_exp_loss.shape)

			ski_expect_loss = np.sum(ski_exp_loss,axis=0)
			ski_cumul_loss += ski_expect_loss[0]

			#cumul_ski_loss += m
			#ski_exp_loss = np.array(ski_exp_loss)
			loss += np.dot(p_t,ski_expect_loss) # Cumulative loss
			#loss = m[x_exp]
			#loss = np.dot(p_t,m)
			#loss = np.amin(m[x_exp])

			# Note that a higher ratio implies a larger loss so we should the decrease the weight of that expert more
			#print(ski_exp_loss.type)
			w_t = w_t * np.exp(-eps*ski_exp_loss[sampled_buyer])

			# Want to update buy expert weights in a better way because i am not being told the true buy cost

			#b_w_t = b_w_t * np.exp(-eta_b*np.absolute((b_pred-b_t)/b_t))
			b_w_t = b_w_t * np.exp(-eta_b*buy_upd)

			m = get_pred_loss_rand(b_t,x_t,b_t,x_pred,lam,num_experts) # Want to compare with if true price told
			cumul_m = cumul_m + m # cumulative loss 
			#cumul_m = m
			min_loss = np.amin(cumul_m)
			#min_loss = cumul_m[0]

			reg_loss1.append(loss-ski_cumul_loss)
			reg_loss2.append(ski_cumul_loss-min_loss)
			regret.append(loss-min_loss)
			#b_diff.append(np.absolute(b_sample-b_t))

		#per_agent_regret.append((cumul_ski_loss-np.amin(cumul_m))/T)
		#lam_regret.append(regret)
		lam_regret[sims] = regret
		#b_regret.append(b_diff)
	avg_lam_regret.append(np.mean(lam_regret,axis=0))

#std_dev = np.std(lam_regret,axis=0)

#p_t = w_t/np.sum(w_t) # Probability distribution over the buy experts
#for i in range(num_bins) :
#	print(np.sum(p_t[bin_ind[i]]))

#print(add)

#print("Standard Deviation : " + str(std_dev[-1]))

#fig,ax = plt.subplots()
#ax.errorbar(range(1,T+1),np.mean(lam_regret,axis=0),yerr=std_dev)
#ax.set_xlabel('Time')
#ax.set_ylabel('Cumulative Regret')
#plt.title(r'Regret Variance : ski-experts=50,$1 \leq \eta_{s} \leq 100$')
#plt.grid(True)
#plt.show()

######## Main Plot Code ############

for j in range(len(avg_lam_regret)) :
	plt.plot(range(1,T+1),avg_lam_regret[j])
#plt.legend([r'$1 \leq \sigma_{x} \leq 50$',r'$51 \leq \sigma_{x} \leq 100$',r'$101 \leq \sigma_{x} \leq 150$'])
#plt.legend([r'$1 \leq \gamma_{r} \leq 20$',r'$151 \leq \gamma_{r}  \leq 160$',r'$251 \leq \gamma_{r} \leq 270$',r'$91 \leq \eta_{s} \leq 110$'])
plt.legend([r'buy-experts=5',r'buy-experts=10',r'buy-experts=50',r'ski-experts=100'])
plt.xlabel('Time')
plt.ylabel('Cumulative Regret')
plt.title(r'num-experts=$15$.$\lambda = 0.6$.$1 \leq \eta_{s} \leq 50$.$1 \leq \gamma_{r} \leq 50$')
#plt.legend([r'$1 \leq x^{t} \leq 50$',r'$201 \leq x^{t} \leq 250$',r'$401 \leq x^{t} \leq 450$',r'$601 \leq x^{t} \leq 650$'])
#plt.legend([r'$\lambda = 0.2$',r'$\lambda = 0.4$',r'$\lambda = 0.6$',r'$\lambda = 0.8$'])
plt.grid(True)
plt.show()

###################################


#for j in range(len(b_regret)) :
#	plt.plot(range(1,T+1),b_regret[j])
#plt.legend([r'$\lambda = 0.5$',r'$\lambda = 0.4$',r'$\lambda = 0.6$',r'$\lambda = 0.8$'])
#plt.xlabel('Time')
#plt.ylabel('Buy Cost Absolute Difference')
#plt.title(r'ski-experts=$100$.buy-experts=100.$50 \leq \gamma_{r} \leq 100$.$1 \leq \eta_{s} \leq 100$.$\lambda = 0.6$')
#plt.legend([r'1 \leq x^{t} \leq 50',r'201 \leq x^{t} \leq 250',r'401 \leq x^{t} \leq 450',r'601 \leq x^{t} \leq 650'])
#plt.grid(True)
#plt.show()

#fig = plt.figure()
#ax = fig.add_subplot(111,projection='3d')

#xs = overlap
#ys = sigmas
#ys = np.ones(num_experts)
#for i in range(num_bins) :
#	# Genreate the x-axis and y-axis coordinate for each expert
#	ys[bin_ind[i]] = sigmas[i]	

#mark = ['o','^','<','v','1','2']
#for i in range(len(init)) :
#	xs = np.ones(num_experts)*overlap[i]
#	zs = per_agent_regret[i]
#	ax.scatter(xs,ys,zs,marker=mark[i])

#ax.set_xlabel(r'Range Scale($i*B_{max} \leq x^{t} \leq (i+1)*B_{max}$)')
#ax.set_xlabel(r'Number of Experts')
#ax.set_ylabel('Error Standard Deviation')
#ax.set_zlabel('Regret over T rounds')

#plt.title(r'$B_{min}=1, B_{max}=50$')
#plt.show()
