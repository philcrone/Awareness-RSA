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

