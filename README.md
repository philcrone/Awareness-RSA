# Awareness-RSA

Rational Speech-Act (RSA) models for speakers sending uninformative messages with addressees who exhibit unawareness. AwarenessRSA.py has an implementation of the model presented at the [2016 NASSLLI student session](http://www.stanford.edu/~pcrone/Documents/NASSLLI-Presentation.pdf). AwarenessRSAGraphs.py uses [matplotlib](http://matplotlib.org) to produce graphs showing the likelihood that a pragmatic speaker would send various messages given different assumptions about the listener.

For other Python implementations of RSA models, see Chris Potts's [pypragmods repository](https://github.com/cgpotts/pypragmods).

## AwarenessRSA.py

A literal speaker, a pragmatic listener, and a pragmatic speaker  are all implemented. Each of these three agents can be simulated from the command line as follows:

```
python AwarenessRSA.py -a <agent-type> -s <speaker-beliefs> -l <listener-beliefs> -w <listener-awareness>
```

Allowable arguments for agent types are `s0` for a literal speaker, `l1` for a pragmatic speaker and `s1` for a pragmatic listener. All other arguments must be a value between 0 and 1.

If the agent is a speaker, the argument to `-s` sets the speaker's subjective probability that `p` is true. The arguments to `-l` and `-w` set the speaker's beliefs about the listener's subjective probability that `p` is true and the speaker's subjective probability that the listener is aware of `p`, respectively.

If the agent is a listener, the argument to `-s` is ignored. The argument to `-l` sets the listener's subjective probability that `p` is true. If the argument to `-w` is greater than or equal to 0.5, the listener is aware of `p`. Otherwise, the listener is unaware of `p`.

More fine-grained control over the speakers and listeners (i.e. changing the speaker's rationality parameter, changing the type of distribution to represent the speaker's beliefs about the listener's beliefs about the world, etc.) is not accessible through the command line.

## AwarenessRSAGraphs.py

Pragmatic speakers reasoning about different types of pragmatic listeners are implemented. The default settings assume a speaker that assigns `p` probability 0.9 and who reasons about pragmatic speakers whose beliefs in `p` and whose likelihood of being aware of `p` take values in the interval [0.05,0.95] by 0.05 increments.

The following can be run from the command line:

```
python AwarenessRSAGraphs.py -m <message-name> -o <output-filename> -i <input-filename>
```

The argument for `-m` must be one of the following given the default settings for AwarenessRSA.py: `p`, `not-p`, `uninform-p`, `uninform-not-p`, `null`. This argument determines which message's probabilities will be graphed.

If an argument to `-o` is provided, the results of the speaker simulations will be saved as a pickled dictionary to the designated filename. If an argument to `-i` is provided, the designated filename provides a pickled dictionary used for the graph.

It may take a while to implement all the speakers given the default settings. If values for these speakers have been pre-computed, you can use the `-i` option (and no `-o` option) to skip the simulation step and just use the pre-computed values. For example, the following will produce a graph showing the likelihood of a speaker sending the message `p` given various properties of the listener, based on the pre-computed values in `sample_posteriors.pickle`.

```
python AwarenessRSAGraphs.py -m p -i sample_posteriors.pickle
```