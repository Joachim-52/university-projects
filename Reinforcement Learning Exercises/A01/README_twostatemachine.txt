Markov Chain Warm up
--------------------

In this exercise you will implement a very simple two state Markov Chain.

Instructions
^^^^^^^^^^^^
1. Copy the file `template-twostatemachine.py` to `twostatemachine.py`
2. Read and understand the code

   - mdp.py :: This file defines an abstract class providing a general interface
     for Markov Decision Processes. No need to edit.
   - twostatemachine.py :: This defines a class implementing the simple
     machine described in `Task 2 - TwoStateMachine`

TODOs
^^^^^
1. Complete the rewards dictionary.
2. Complete the probabilities dictionary.
3. Complete the `successor_states` method.
4. Add analytic solution at the end
Testing
^^^^^^^

- `python twostatemachine.py` :: a basic example of usage
- `python test_twostatemachine.py` :: runs a few unit tests.
-  Add your analytical solution to Testcase 5 if you want to compare.

Good luck!
