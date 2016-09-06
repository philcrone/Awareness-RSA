import sys
import getopt
import numpy as np
import math
import itertools
import time
import copy
import pp
import pickle
import inspect
import scipy.stats as stats
from scipy.linalg import norm 
from sklearn.preprocessing import normalize
from sklearn.utils.extmath import cartesian

worlds = np.array(['w1','w2'])
"""
There are only two worlds in this model. The proposition p corresponds to {w1}.
"""

awareness_states = np.array(['aware','unaware'])
"""
The listener has two potential awareness states, corresponding to the listener being aware
of p/attending to p or not being aware/not attending to p.
"""

belief_threshold = 0.85
"""
Threshold relevant for evaluating uninformative messages, e.g. 'As you know, p.' Such 
messages will be true so long as p is true and subjective probability assigned to p by 
listener exceeds the above threshold.
"""

doxastic_weight = 0.15

informativity_weight = 5

"""
The above two weights determine how the utilities assigned to messages by a pragmatic
speaker depend upon informativity and the distance between the speaker's prior beliefs
about the listener and the listener's posterior beliefs about the speaker's beliefs about
the listener.
"""

class Message:
	def __init__(self, name = None, sem = None, cost= None):
		self.name = name
		self.sem = sem
		self.cost = cost
	
	def get_name(self):
		return self.name
	
	def get_sem(self):
		return self.sem
		
	def get_cost(self):
		return self.cost
		
	def set_cost(self, c):
		self.cost = c
		
p = Message(name = 'p',sem = lambda w, l: 1 if w in ['w1'] else 0, cost=.2)

"""
Message p is true in world w1 and has cost of 0.2.
"""

not_p = Message(name = 'not-p',sem = lambda w, l: 1 if w in ['w2'] else 0, cost = .2)

"""
Message not-p is true in world w2 and has cost of 0.2.
"""

null = Message(name = 'null', sem = lambda w, l: 1 if True else 0, cost = 0)

"""
Message null is true in all worlds and has cost of 0.
"""

uninform_p = Message(name = 'uninform-p', sem = lambda w, l: 1 if w in ['w1'] and l > belief_threshold else 0, cost = .4)

"""
Message uninform-p is true in w1 if the listener's subjective belief in w1 exceeds the
relevant treshold and has cost of 0.4.
"""

uninform_not_p = Message(name = 'uninform-not-p', sem = lambda w, l: 1 if w in ['w2'] and l > belief_threshold else 0, cost = .4)

"""
Message uninform-not-p is true in w2 if the listener's subjective belief in w2 exceeds 
the relevant treshold and has cost of 0.4.
"""

messages = np.array([p,not_p,null,uninform_p,uninform_not_p])
		
def normalize(dict):
	"""
	Function to normalize a dict. Output is a dictionary with the same keys as the
	input, but values sum to 1.
	"""
	return {i: j / sum(dict.values()) for i, j in dict.items()}
	
def params(mu,var):
	"""
	Function to generate parameters for a beta distribution given mean and variance.
	"""
	alpha = ((1 - mu) / var - 1 / mu) * mu ** 2
	beta = alpha * (1 / mu - 1)
	return alpha, beta
	
def hellinger(p, q):
	"""
	Function to calculate the Hellinger distance between two discrete probability 
	distributions. Hellinger distance is the only distance between two probability
	distributions used in this model, but in principle other measures could be used.
	"""
	return norm(np.sqrt(p) - np.sqrt(q)) / np.sqrt(2)
    
def quick_literal_speaker(speaker_beliefs, listener_beliefs):
	"""
	Rather than build a full Speaker object to simulate a literal speaker, this 
	function serves as a faster way to get a literal speaker's posterior distribution over
	messages. Inputs should be a numeric value representing the speaker's subjective
	probability of w1 and a probability distribution (a stats.rv_continuous object) 
	representing the speaker's beliefs about the listener's subjective probability of w1.
	""" 
	posteriors = {	'p':speaker_beliefs,
					'not-p':1-speaker_beliefs,
					'null':1,
					'uninform-p': speaker_beliefs * (1 - listener_beliefs.cdf(belief_threshold)),
					'uninform-not-p': (1-speaker_beliefs) * (1 - listener_beliefs.cdf(1 - belief_threshold))}
	posteriors = normalize(posteriors)
	return posteriors
    
beta_dists = {}
for val in np.arange(0.05,1,0.05):
	beta_dists[val] = stats.beta(*params(val,0.01))

"""
Generate beta distributions with means ranging from 0.05 to 0.95 with variance 0.01. These
will be used by listeners and speakers below, and it's more efficient to pre-generate
these distributions than generate them as needed.
"""
	
quick_pos = {}
grid = np.arange(0.05,1,0.05)
for m in messages:
	quick_pos[m.get_name()] = {}
	for pair in itertools.product(grid,grid):
		quick_pos[m.get_name()][pair] = quick_literal_speaker(pair[0],beta_dists[pair[1]])[m.get_name()]

"""
Generate posteriors over messages for different literal speakers. In particular, we 
generate literal speakers whose subjective probability of w1 can take any value in
[0.05,0.95] in 0.05 increments and whose beliefs about the listener's subjective
probability of w1 are given by a beta distribution with variance 0.01 and whose mean can
take any value [0.05,0.95] in 0.05 increments.
"""

class Listener:
	def __init__(self, 
			priors = None, 
			awareness = None, 
			speaker = None,
			quick_speaker=False):
		"""
		Listener is initialized with priors, an awareness state, and a speaker about
		which the listener is reasoning. A listener initialized with no speaker is a
		literal listener. If quick_speaker is True, listener is a pragmatic listener
		reasoning about a literal speaker, but literal speaker is modeled using
		quick_literal_speaker rather than a full Speaker object.
		"""
		self.priors = priors
		self.awareness = awareness
		self.speaker = speaker
		self.quick_speaker = quick_speaker
		self.posteriors = self.priors
		
	def get_priors(self):
		return self.priors
		
	def get_awareness(self):
		return self.awareness
		
	def get_speaker(self):
		return self.speaker

	def get_posteriors(self):
		return self.posteriors
		
	def set_priors(self,priors):
		self.priors = priors
		
	def set_awareness(self,awareness):
		self.awareness = awareness
		
	def compute_posteriors(self, m):
		"""
		Calculate posterior beliefs given a message m.
		"""
		self.posteriors = {}
		if self.speaker or self.quick_speaker:
			"""
			Check if listener is a pragmatic listener.
			"""
			if not self.quick_speaker and self.speaker.get_listener():
				"""
				Currently not implemented: pragmatic listener that reasons about a 
				pragmatic speaker.
				"""
				pass			
			else:
				if m != null:
					"""
					Listener only does pragmatic reasoning if the message is not null. 
					This condition can be changed if we'd like to make alternative 
					assumptions. For example, we could have the listener do pragmatic 
					reasoning unless the message is null AND the listener is unaware.
					"""
					grid = np.arange(0.05,1,0.05)
			 		for pair in itertools.product(grid,grid):
						if self.quick_speaker:
							self.posteriors[pair] = quick_pos[m.get_name()][pair]
						else:
		 					self.speaker.set_priors( {	'worlds':[pair[0],1-pair[0]],
			 										'listener-worlds':stats.beta(*params(pair[1],0.01)),
			 										'listener-awareness':self.speaker.get_priors()['listener-awareness']})
		 					self.speaker.compute_posteriors()
		 					self.posteriors[pair] = (self.speaker.get_posteriors()[m.get_name()] * 
			 									self.priors['speaker-worlds'].pdf(pair[0]) *
			 									self.priors['speaker-listener-worlds'].pdf(pair[1]))
			 	
			 		self.posteriors = normalize(self.posteriors)
			 		
			 		"""
			 		Above, we simulate literal speakers for a range of values for the 
			 		speaker's beliefs about the world and the speaker's beliefs about the 
			 		listeners's beliefs about the world. Find the probability that the 
			 		speaker would have sent the message m and then normalize across all 
			 		speakers considered.
			 		"""
			 		
			 		grid_probs = np.zeros(19)
			 		
			 		for key in self.posteriors.keys():
			 			grid_probs[int(key[1]*20 - 1)] += self.posteriors[key]
			 				
			 		data = np.random.choice(grid,500,p=grid_probs)
			 		
			 		fitted_params = stats.beta.fit(data,floc=0,fscale=1)
			 		
			 		"""
			 		Fit a beta distribution giving the likelihood of the speaker's
			 		beliefs about the listener's beliefs about the world.
			 		"""
			 					 		
			 		speaker_mean = 0.
			 		for key in self.posteriors.keys():
			 			speaker_mean += key[0] * self.posteriors[key]
			 			
			 		"""
			 		Compute the weighted mean of the speaker's beliefs about the world.
			 		We assume here that the listener has full confidence in the speaker 
			 		and will therefore adopt the speaker's beliefs about the world  as the 
			 		listener's posterior beliefs about the world.
			 		"""
			 		
					self.posteriors['worlds'] = [speaker_mean, 1 - speaker_mean]
			 		self.posteriors['speaker-listener-worlds'] = stats.beta(fitted_params[0],fitted_params[1],loc=0,scale=1)
			 		
			 		"""
			 		Update the listener's posteriors posteriors.
			 		"""

			 	else:
			 		"""
			 		If the listener receives the null message, no prgamatic reasoning
			 		occurs. Therefore, posteriors will be the same as priors.
			 		"""
			 		self.posteriors = self.priors
		else:
			"""
			Currently not implemented: listener listener. Implementing a listener with
			uninformative messages is non-trivial. Suppose, for example, a speaker sends 
			the message uninform-p when the listener does not, in fact, believe that p is 
			the case. The speaker's message is false, which opens the question of whether 
			the listener should update at all or update as if the speaker had sent message p.
			"""
			pass
	
class Speaker:
	def __init__(self, 
			priors = None,
			alpha = None,
			listener = None):
		"""
		Speaker is initialized with priors, a rationality parameter alpha, and a
		listener. A speaker initialized with no listener is a literal speaker.
		"""
		self.priors = priors
		self.alpha = alpha
		self.listener = listener
		self.utilites = {}
		self.posteriors = None
		
	def get_priors(self):
		return self.priors
		
	def get_alpha(self):
		return self.alpha
		
	def get_listener(self):
		return self.listener
		
	def get_utilities(self):
		return self.utilities
		
	def get_posteriors(self):
		return self.posteriors
		
	def set_priors(self,new_priors):
		self.priors = new_priors
	
	def compute_utilities(self):
		"""
		Compute the utilities associated with each message. The speaker is assumed to have
		three goals: ensure that the listener is aware of p, inform the listener about the
		truth-value of p, inform the listener about the speaker's beliefs abou the
		listener.
		"""
		utilities = {}
				
		for m in messages:
			utilities[m.get_name()] = 0.
			for a in awareness_states:
				"""
				The speaker calculates the utility of each message for each possible 
				awareness state of the listener, then weights these utilites by the
				subjective probability the speaker assigns to each awareness state.
				"""
				if m == null and a == 'unaware':
					"""
					If the listener is unaware, then the speaker fails to achieve the goal
					of making the listener aware of p. Here, this failure is represented
					by a constant, negative value associated with leaving the listener
					unaware. 
					
					Note that in the current implementation, a null message with an
					unaware listener is not more costly if the listener has false beliefs
					about the world or about the speaker's beliefs about the listener.
					This should probably be changed in future versions of the model.
					"""
					util = -5
				else:
					self.listener.set_awareness(a)
					self.listener.compute_posteriors(m)
					
					l_posteriors = self.listener.get_posteriors()
					util = 0.
			
					self_data = [stats.beta(*params(self.listener.get_priors()['worlds'][0],0.001)).pdf(i * 0.01) for i in xrange(1,100)]
					listener_data = [l_posteriors['speaker-listener-worlds'].pdf(i * 0.01) for i in xrange(1,100)]	
					
					"""
					Above, we generate two discrete probability distributions, the first
					representing the speaker's beliefs about the listener's beliefs about
					the world and the second respresenting the listener's beliefs about
					the speaker's beliefs about the listener's beliefs about the world.
					"""
			
					util -= (m.get_cost() + (doxastic_weight * hellinger(self_data,listener_data)))
					
					"""
					Subtract from the utility the message cost and the distance between 
					the two discrete probability distributions generated. The distance is
					weighted by doxastic_weight.
					"""
			
					util += informativity_weight * np.log(l_posteriors['worlds'][0]) * self.priors['worlds'][0]
			
					util += informativity_weight * np.log(l_posteriors['worlds'][1]) * self.priors['worlds'][1]
					
					"""
					Add to the utility the log of the listener's posterior beleif in each
					world given the message m, weighted by the speaker's prior belief in
					the world and by informativity_weight.
					"""

				if a == 'aware':
					util *= self.priors['listener-awareness'][0]
				else:
					util *= self.priors['listener-awareness'][1] 
					
				"""
				Utilities were calculated given each awareness state. We now weight these
				values by the speaker's prior probability of each awareness state.
				"""
									
				utilities[m.get_name()] += util
		
		self.utilities = utilities
		
	def compute_posteriors(self):
		posteriors = {}
		if self.listener:
			"""
			If the speaker is a pragmatic speaker, posterior probabilites for each message
			are calculated by using a soft-max rule. As self.alha goes to infinity, the
			likelihood of sending the message that maximizes expected utility increases.
			"""
			self.compute_utilities()
			for m in messages:
				posteriors[m.get_name()] = math.exp(self.alpha * self.utilities[m.get_name()])
				
		else: 
			"""
			If the speaker is a literal speaker, posterior probabilities for each message
			are proportional to the likelihood that the message is true based on the
			speaker's priors.
			"""
			for m in messages:
				posteriors[m.get_name()] = 	(m.sem('w1',1) * 
											self.priors['worlds'][0] * 
											(1 - self.priors['listener-worlds'].cdf(belief_threshold)) +
											m.sem('w1',0) * 
											self.priors['worlds'][0] * 
											self.priors['listener-worlds'].cdf(belief_threshold) +
											m.sem('w2',1) * 
											self.priors['worlds'][1] * 
											self.priors['listener-worlds'].cdf(1 - belief_threshold) +
											m.sem('w2',0) * 
											self.priors['worlds'][1] * 
											(1 - self.priors['listener-worlds'].cdf(1 - belief_threshold)))
		
		posteriors = normalize(posteriors)
			
		self.posteriors = posteriors
		
def main(argv):
	agent = 's1'
	sb = 0.9
	lb = 0.5
	la = 0.5
	try:
		opts, args = getopt.getopt(argv,"ha:s:l:w:",["agent=","speaker-belief=","listener-belief=","listener-awareness="])
	except getopt.GetoptError:
		print 'Awareness-RSA-NASSLLI.py -a <agent-type> -s <speaker-beliefs> -l <listener-beliefs> -w <listener-awareness>'
		sys.exit(2)
	for opt, arg in opts:
		if opt == '-h':
			print 'Awareness-RSA-NASSLLI.py -a <agent-type> -sb <speaker-beliefs> -lb <listener-beliefs> -la <listener-awareness>'
			sys.exit()
		elif opt in ("-a", "--agent"):
			agent = arg
		elif opt in ("-s", "--speaker-belief"):
			try:
				sb = float(arg)
			except ValueError:
				print 'Speaker beliefs must be a value between 0 and 1.'
				sys.exit(2)
		elif opt in ("-l", "--listener-belief"):
			try:
				lb = float(arg)
			except ValueError:
				print 'Listener beliefs must be a value between 0 and 1.'
				sys.exit(2)
		elif opt in ("-w", "--listener-awareness"):
			try:
				la = float(arg)
			except ValueError:
				print 'Listener beliefs must be a value between 0 and 1.'
				sys.exit(2)
	if agent == 's0':
		speaker_0 = Speaker(	priors = {	'worlds':np.array([sb,1-sb]),
									'listener-worlds':stats.beta(*params(lb,0.01)),
									'listener-awareness':[la,1-la]},
						alpha = 5,
						listener = None )
		speaker_0.compute_posteriors()
		print 'Literal speaker posteriors over messages:'
		print speaker_0.get_posteriors()
 	elif agent == 'l1':
 		if la > 0.5:
 			la = 'aware'
 		else:
 			la = 'unaware'
 		listener_1 = Listener(	priors = {	'worlds':np.array([lb,1-lb]),
											'speaker-worlds':stats.uniform(),
											'speaker-listener-worlds':stats.uniform()},
						awareness = la,
						speaker = None,
						quick_speaker = True )
		listener_1.compute_posteriors()
		print 'Pragmatic listener posteriors over worlds given each message:'
		for m in messages:
			print m.get_name()
			print listener_1.get_posteriors(m.get_name()) 
 	elif agent == 's1':
 		listener_1 = Listener(	priors = {	'worlds':np.array([lb,1-lb]),
									'speaker-worlds':stats.uniform(),
									'speaker-listener-worlds':stats.uniform()},
						awareness = None,
						speaker = None,
						quick_speaker = True )

		speaker_1 = Speaker(	priors = {	'worlds':np.array([sb,1-sb]),
									'listener-awareness':[la,1-la]},
						alpha = 5,
						listener = listener_1 )
		speaker_1.compute_posteriors()
		print 'Pragmatic speaker posteriors over messages:'
		print speaker_1.get_posteriors()
 	else:
 		print "Agent must be either 's0', 'l1', or 's1'."
	
	
if __name__ == '__main__':
	main(sys.argv[1:])
