# Random Walk
The human eye is never perfectly still. Even when looking at just one point, the center of the visual field is shifted around on the retina by a jittery drift movement. This drift has a few interesting statistical properties, speed of movement, preference for horizontal over vertical motion, and self-avoidance.

We are going to build a simplified version of a eye movement drift model! We will (git) stash the self-avoidance for now, and focus just on making a walker that moves across a field over time. At each time step, or iteration, the walker choses the next position. For this choice we assume it uses two pieces of information:
1. it can only move a limited distance in a single particular timestep. The step distribution is a two dimensional Gaussian around the walker's current position- it prefers to move to a nearby/adjacent gridpoint and has a very low probability of jumping far away.
2. The walker is also influenced by the activation landscape that it is on. If the activation landscape is flat, only 1. matters. If the activation landscape has a gradient, it walks toward the minimum/maximum.


## Exercises
1. Turn functions into class. Write a grid search.
-> learning: how to write a class

2. Write read parameters from json file
-> learning: how to make reproducible experiments

3. 
    a) Give snippets for how the activation is generated - give them code to visualize for fun. they write the functions for initialization within the class. The initialize function then takes a string and uses if/else
        -> discuss whether this generalizes well
   b) break out initializer of the field into a new class.
        -> learning:  breaking out varying stuff

4. write code to pass in estimation algorithm ( grid search)  run estimation of best fitting parameters given some precomputed data. Give w different estimation algorithm.
-> learning:  breaking out varying stuff, but with less guidance

5. simulate a trajectory. Save alongside the initialization json. run notebook that loads the data and makes some plots.  (can another team reproduce the same plots?)
-> structure of experiments