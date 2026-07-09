# Introduction

This section was written befoer a full rewrite was performed on sections 1 to 3, therefore there are a few sections that need to be updated. Your task is going to be both to write a proper experimental section and explanation of the reasoning behind each of the experiments using both the text that is currently written in this sections and the previous updated sections. To do so you can ask questions. The code has been updated to a python based code that uses gurobi too, the C++ code has been discarded. 

# Experiments

We will revise a little bit the experiments that needs to be proposed and explained in this section. Let's start by the simple things, we will begin with the experiments that come from the Doer and Clement papers. They propose using the following values for N: 0, 50, 100, 180, 220, 260, 350, 420, 500. Therefore, we will run our experiments with this values of N. The configuration used for all the experiments is the following: 

pipeline:
  population_size: 200
  selection_size: 20 # percentage of population (e.g. 10 = 10%)

run:
  generations: 500
  epsilon: 0.0001
  num_workers: null

pbil:
  learning_rate: 0.1
  mutation_rate: 0.01

birkhoff:
  alpha: 0.1

sinkhorn:
  iterations: 10


As running all this experiments with all the methods with such a modest hardware is relatively hard we will explore the best methods with N = 20 and from this recollect the best configuration and method that later will be applied to the whole N range. The experiments are the followign: 

1. Birkhoff vs PBIL: It compares the performance and the results obtained between PBIL and Birkhoff, we expect the Birkhoff algorithm to be more explorative and have stronger spikes up and down. While we expect the PBIL to converge slower with fewer exploration. 
2. Birkhoff with warm start vs cold start: Explore how the Birkhoff methods behaves under cold and warm start. As the warm start introduces a bias for the EDA pipeline we expect the cold start to find better solutions but to take more time to converge and the warm start to converge faster though as it has this bias it may find worse solutions. 
3. PBIL with warm start vs cold start: Explore how the PBIL methods behaves under cold and warm start. As the warm start introduces a bias for the EDA pipeline we expect the cold start to find better solutions but to take more time to converge and the warm start to converge faster though as it has this bias it may find worse solutions. 
3. PBIL with SVD noise injecion: Explore how the PBIL methods behaves under cold and warm start. As the warm start introduces a bias for the EDA pipeline we expect the cold start to find better solutions but to take more time to converge and the warm start to converge faster though as it has this bias it may find worse solutions. 
3. Birkhoff with SVD noise injecion: Explore how the PBIL methods behaves under cold and warm start. As the warm start introduces a bias for the EDA pipeline we expect the cold start to find better solutions but to take more time to converge and the warm start to converge faster though as it has this bias it may find worse solutions. 


