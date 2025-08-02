Dear everyone,

please find enclosed my attempt to produce a modular, heavy-duty, easy-to-use-in-other-programs implementation of a combination of mine and Giuliana's bisimulation minimization [1] plus Markov state lumping [2] algorithms. This is the API I have promised to write and I hope you can relatively easily integrate it to your programs and run tests.

Please start by reading the long first comment in MDPmin.cc. Depending on what you do, you will eventually need at least err.cc and MDP_minimize.cc, but not necessarily the rest.

Designing the API was much more challenging than I thought, but I have tested this solution both conceptually (by using it also in another program) and that it computes correct outputs.

Please send me feedback when you have the chance.

-- Antti

1. Antti Valmari
   Simple Bisimilarity Minimization in O(m log n) Time
   Fundamenta Informaticae 105 3, 319--339
   2010

2. Antti Valmari, Giuliana Franceschinis
   Simple O (m log n) time Markov chain lumping
   International Conference on Tools and Algorithms for the Construction and Analysis of Systems
   Lecture Notes in Computer Science 6015, 38--52
   Springer 2010
