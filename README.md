# FlowMatchNN

This repo is for the package and experiment of the paper Flow Matching with Neural Networks

We need: 
1. Implement the package that the flow matching with NN method as the paper write (Alg 1 and Alg 2)

2. We need to run the experiments on the following: 

(1) We need to plot a figure for the loss function (vertical axis) vs t (horizontal axis). Here is for each time step t in alg 2. 

(2) We need to plot a figure for the TV-distance (vertical axis) vs t (horizontal axis). Here is for each time step t in alg 2. 

The info and parameter values are: (but you need to make sure that they can be udpated easily. you need to have a argparse function to load all args when you are going to run the exp. You also need to write shell scripts to submit the job to cluster)
In alg1: 
n = 100 (sample size)
d = 5 (sample dimension)
x1 in alg 1 to be N(0, 1)
sigma(tau) = tau; mu(tau) = 1-tau

In alg2: 
kappa = 1
T = 200