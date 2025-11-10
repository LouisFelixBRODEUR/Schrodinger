# Schrodinger
This Python code is in support of the GPH-3004 Quantum Mechanics class for Engineers.

Daniel Côté dccote@cervo.ulaval.ca

## Getting started

 A set of classes to solve the Schrödinger equation for various potentials.

 Use `pip install -r requirements.txt` to install the required packages.



## What is this?

Four scripts:

1. `schrodinger.py`: classes to solve the schrodinger equation in various potentials. Several potentials are already programmed:

   1. Infinite well
   2. Finite well
   3. Harmonic potential
   4. Half-harmonic potential

2. `bells_inequalities.py` : Following Chapter 4 in McIntyre, an implementation of what the results would be in a Bell's experiment with entangled particles.  The code will produce the results as would be expected if Einstein-Podolsky-Rosen were correct and hidden variables exist, or if the current mathematical formalism with the Copenhagen interation is sufficient to explain the results.

   1. The code will produce two sets of correlations which are different : never better than 44.4% with EPR, and 50% with QM.  Only an experiment can decide.
   2. Experiments were performed and showed that QM is correct.

3. `deriveintegrate.py `: Differential operators can be represented by matrices.  In suggests that integral operators can alos be represented as the inverse of differential operators.  What may come as a surprise is that the definition of an operator does not set its representation unambiguously: we must also consider the domain of application, which includes the boundary conditions of the vectors.

   Simple cases are described to demonstrate this.

4. `testschrodinger.py` because we always test our code.





