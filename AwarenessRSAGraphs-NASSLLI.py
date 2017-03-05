import sys
import getopt
import pickle
import itertools
import numpy as np
import AwarenessRSA-NASSLLI as arsa
import scipy.stats as stats
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors as mcolors
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt

pos = {}

def get_values(filename = None):
	"""
	Function that returns probability of sending each message for a pragmatic speaker that
	assigns p probability 0.9. The speaker reasons about pragmatic listeners with beliefs
	in the interval [0.05,0.95] and who is aware with likelihoods in the interval
	[0.05,0.95]. If filename is provided, results will be saved as a pickled dictionary
	to given filename.
	"""
	if filename:
		file = open(filename,'w')
	grid = np.arange(0.05,1,0.05)

	for pair in itertools.product(grid,grid):
		b = pair[0]
		a = pair[1]
		listener_1 = arsa.Listener(	priors = {	'worlds':np.array([b,1-b]),
									'speaker-worlds':stats.uniform(),
									'speaker-listener-worlds':stats.uniform()},
						awareness = 'aware',
						speaker = None,
						quick_speaker = True )

		speaker_1 = arsa.Speaker(	priors = {	'worlds':np.array([0.9,0.1]),
									'listener-awareness':[a,1-a]},
						alpha = 5,
						listener = listener_1 )
		for i in xrange(0,10):
			speaker_1.compute_posteriors()
			local_pos = speaker_1.get_posteriors()
			if pair in pos.keys():
				for key in pos[pair]:
					pos[pair][key] += local_pos[key]
			else:
				pos[pair] = local_pos

		pos[pair] = arsa.normalize(pos[pair])

	pickle.dump(pos,file)

def make_colormap(seq):
    """
    Return a LinearSegmentedColormap seq: a sequence of floats and RGB-tuples. The floats
    should be increasing and in the interval (0,1). Source:
    http://stackoverflow.com/questions/16834861/create-own-colormap-using-matplotlib-and-plot-color-scale.
    """
    seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
    cdict = {'red': [], 'green': [], 'blue': []}
    for i, item in enumerate(seq):
        if isinstance(item, float):
            r1, g1, b1 = seq[i - 1]
            r2, g2, b2 = seq[i + 1]
            cdict['red'].append([item, r1, r2])
            cdict['green'].append([item, g1, g2])
            cdict['blue'].append([item, b1, b2])
    return mcolors.LinearSegmentedColormap('CustomMap', cdict)

def diverge_map(high=(0.471, 0.082, 0.082), low=(0.035,0.443,0.698)):
    """
    Low and high are colors that will be used for the two ends of the spectrum. They can
    be either color strings or rgb color tuples. Source:
    http://stackoverflow.com/questions/16834861/create-own-colormap-using-matplotlib-and-plot-color-scale.
    """
    c = mcolors.ColorConverter().to_rgb
    if isinstance(low, basestring): low = c(low)
    if isinstance(high, basestring): high = c(high)
    return make_colormap([low, c('white'), 0.5, c('white'), high])

def graph(m,filename):
	"""
	Produce graphs showing the likelihood of a pragmatic speaker sending message m given
	different beliefs about listener's beliefs in the world and beliefs about the
	listener's awareness state. If filename is provided, pickled dictionary at given
	location will be used.
	"""

	new_pos = {}

	if filename:
		file = open(filename, 'rb')
		pos = pickle.load(file)
	for key in pos.keys():
		index1 = key[0]*20 - 1
		index2 = key[1]*20 - 1
		new_pos[(index1, index2)] = pos[key]

# 	print new_pos

	fig = plt.figure()
	ax = fig.gca(projection='3d')
	X = np.arange(0.05,1,0.05)
	Y = np.arange(0.05,1,0.05)
	X, Y = np.meshgrid(X, Y)
	Z = np.full((19,19),0.)

	for pairs in [(x,y) for x in X[0] for y in X[0]]:
		index1 = pairs[0]*20 - 1
		index2 = pairs[1]*20 - 1
		Z[index1][index2] = new_pos[(index1,index2)][m]

	surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=diverge_map(),
						   linewidth=0, antialiased=False)
	ax.set_zlim(0, 1.0)

	ax.yaxis._axinfo['label']['space_factor'] = 2
	ax.xaxis._axinfo['label']['space_factor'] = 2
	ax.zaxis._axinfo['label']['space_factor'] = 2

	ax.zaxis.set_major_locator(LinearLocator(11))
	ax.xaxis.set_major_locator(LinearLocator(11))
	ax.yaxis.set_major_locator(LinearLocator(11))
	ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

	ax.set_xlabel(r"$P_{S_2}(a_l)$",fontsize=16)
	ax.set_ylabel(r"$\mu(P_{S_2}(b_l))$",fontsize=16)
	ax.set_zlabel(r'$P_{S_2}(m)$',fontsize=16)

	plt.show()

def main(argv):
	m = 'p'
	i = None
	o = None
	try:
		opts, args = getopt.getopt(argv,"hm:i:o:",["message=","input=","output="])
	except getopt.GetoptError:
		print 'Awareness-RSA-NASSLLI-graphs.py -m <message-name> -i <input-filename> -o <output-filename>'
		sys.exit(2)
	for opt, arg in opts:
		if opt == '-h':
			print 'Awareness-RSA-NASSLLI-graphs.py -m <message-name> -i <input-filename> -o <output-filename>'
			sys.exit()
		elif opt in ("-m", "--message"):
			m = arg
		elif opt in ("-i", "--input"):
			i = arg
		elif opt in ("-o", "--output"):
			o = arg
	if o:
		get_values(o)
		if i:
			try:
				graph(m,i)
			except IOError:
				'Input must be a valid filename or path.'
				sys.exit(2)
			except TypeError:
				'Input file must be a pickled dictionary.'
				sys.exit(2)
			except KeyError:
				"Either the input file was not a pickled dictionary or the message was not in the dictionary's keys."
				sys.exit(2)
		else:
			graph(m)
	elif i:
		try:
			graph(m,i)
		except IOError:
			'Input must be a valid filename or path.'
			sys.exit(2)
		except TypeError:
			'Input file must be a pickled dictionary.'
			sys.exit(2)
		except KeyError:
			"Either the input file was not a pickled dictionary or the message was not in the dictionary's keys."
			sys.exit(2)
	else:
		get_values()
		graph(m)

if __name__ == '__main__':
	main(sys.argv[1:])
