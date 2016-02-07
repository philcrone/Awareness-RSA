import numpy as np
import math
import itertools

def exp_vec(x):
	"""Vectorized version of exp function."""
	return np.vectorize(math.exp)(x)

def rownorm(mat):
	"""Normalize rows of each two-dimensional sub-matrix within a three-dimensional matrix."""
	subarrs = []
	for subarr in mat:
		for i in xrange(0,len(subarr)):
			if np.sum(subarr[i]) == 0:
				subarr[i] = np.ones(len(subarr[i]))
		new_subarr = np.divide(subarr.T,np.sum(subarr,axis=1)).T
		subarrs.append(new_subarr)
	return np.array(subarrs)

def matnorm(mat):
	"""Normalize each two-dimensional sub-matrix within a three-dimensional matrix."""
	subarrs = []
	for subarr in mat:
		new_subarr = subarr / np.sum(subarr)
		subarrs.append(new_subarr)
	return np.array(subarrs) 
	
def my_log(x):
	"""Allow for cases in which we want to take log(0), which is undefined."""
	if x != 0:
		return math.log(x)
	else:
		return -999.0
		
def log_vec(x):
	"""Vectorized version of logarithm."""
	return np.vectorize(my_log)(x)

def speaker(listener,alpha,c):
	"""Listener is a three-dimensional matrix giving a listener's probability distribution over worlds given a message and a doxastic state. Alpha is a parameter between 0 and infinity that determins how rational the speaker is. C is a vector giving the costs associated with each message.

Speaker returns a three-dimensional matrix giving a pragmatic speaker's probability distributions over messages given a world and listener's doxastic state."""
	
	U = log_vec(listener.T) - c
	
	"""U gives utilities associated with each message given a world and a doxastic state."""
	
	P = exp_vec(alpha * U)
	
	"""P gives probabilities that a speaker will send a message given a world and a doxastic state."""
	
	return rownorm(P)

def listener(speaker):
	"""Speaker is a three-dimensional matrix giving a speaker's probability distribution over messages given a doxastic state and a world. 
	
Listener returns a three-dimensional matrix giving a listener's probability distribution over worlds and doxastic states given a message."""

	return matnorm(speaker.T) 

"""First, an example without clarity assertions. This example is intended to illustrate how a listener could reason about a speaker's beliefs about the listener's doxastic state."""

messages_inform = ['p','not-p','null']

"""First, define the possible messages. A speaker may assert p, its negation, or stay silent."""

costs_inform = [.25,.25,0]

"""Next, define the costs for each message. Assume that asserting p and not-p both have the same cost. The null message has no cost."""

dox_states_inform = ['believe-p','believe-not-p','ignorant']

"""Next, define possible doxastic states for the listener. The listener either believes p, believes not-p, or is ignorant."""

worlds = ['w1','w2']

"""Next, define possible worlds; p is true in w1 and false in w2."""

l0_inform =	np.array([	[	[1.0,0.0],
							[1.0,0.0],
							[1.0,0.0]],						
						
						[	[0.0,1.0],
							[0.0,1.0],
							[0.0,1.0]],
						
						[	[1.0,0.0],
							[0.0,1.0],
							[0.5,0.5]]	])

"""Three-dimensional matrix representing a literal listener's probability distribution over worlds given a message and a doxastic states. Each two-dimensional sub-matrix represents a particular message (p, not-p, null). Within each sub-matrix, rows represent doxastic states (believe-p,believe-not-p,ignorant) and columns represent worlds (w1,w2)."""

messages_aware = ['p','clear-p','not-p','clear-not-p','null']

"""Define possible messages for awareness example."""

costs_aware = np.array([.1,.2,.1,.2,0])

"""Define costs for awareness example."""

dox_states_aware = ['believe-p','unaware-p','believe-not-p','unaware-not-p','ignorant']

"""Define doxastic states for awareness example."""

							
l0_aware =	np.array([	[	[1.0,0.0],
							[1.0,0.0],
							[1.0,0.0],
							[1.0,0.0],
							[1.0,0.0]],	
							
						[	[1.0,0.0],
							[1.0,0.0],
							[0.5,0.5],
							[0.5,0.5],
							[0.5,0.5]],						
						
						[	[0.0,1.0],
							[0.0,1.0],
							[0.0,1.0],
							[0.0,1.0],
							[0.0,1.0]],
							
						[	[0.5,0.5],
							[0.5,0.5],
							[0.0,1.0],
							[0.0,1.0],
							[0.5,0.5]],
						
						[	[1.0,0.0],
							[0.0,1.0],
							[0.0,1.0],
							[1.0,0.0],
							[0.5,0.5]]	])

"""Three-dimensional matrix representing a literal listener's probability distribution over worlds given a message and a doxastic states in the awareness example. Each two-dimensional sub-matrix represents a particular message (p, not-p, null). Within each sub-matrix, rows represent doxastic states (believe-p,believe-not-p,ignorant) and columns represent worlds (w1,w2)."""

def inform_example():
	s1 = speaker(l0_inform,5,costs_inform)
	l1 = listener(s1)
	
	print 'Example without awareness: \n'
	
	for m in messages_inform:
		subarr = l1[messages_inform.index(m)]
		print 'Speaker sends message %s \n' %m
		for w, d in itertools.product(worlds,dox_states_inform):
			print '\t P(%s,%s) = %f' % (w,d, subarr[dox_states_inform.index(d)][worlds.index(w)])
		print '\n'
	
def aware_example():
	s1 = speaker(l0_aware,5,costs_aware)
	l1 = listener(s1)
	
	print 'Example without awareness: \n'
	
	for m in messages_aware:
		subarr = l1[messages_aware.index(m)]
		print 'Speaker sends message %s \n' %m
		for w, d in itertools.product(worlds,dox_states_aware):
			print '\t P(%s,%s) = %f' % (w,d, subarr[dox_states_aware.index(d)][worlds.index(w)])
		print '\n'

if __name__ == '__main__':
	inform_example()
	aware_example()
	