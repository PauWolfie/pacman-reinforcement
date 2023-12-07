# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """

    def __init__(self, mdp, discount=0.9, iterations=100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter()  # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Get the list of states in the Markov Decision Process (MDP)
        states = self.mdp.getStates()

        # Perform value iteration for the specified number of iterations
        for _ in range(self.iterations):
            # Create a copy of current values to store the updated values
            next_values = self.values.copy()

            # Iterate over all states in the MDP
            for state in states:
                # Skip terminal states
                if not self.mdp.isTerminal(state):
                    # Get the recommended action for the current state
                    action = self.getAction(state)

                    # Update the value for the current state using the recommended action
                    next_values[state] = self.computeQValueFromValues(state, action)

            # Update the global values dictionary with the newly computed values
            self.values = next_values


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def computeQValueFromValues(self, state, action):
        """
        Compute the Q-value of action in state from the
        value function stored in self.values.
        """

        # Get the possible successor states and their probabilities
        successors = self.mdp.getTransitionStatesAndProbs(state, action)

        # Initialize the Q-value to zero
        q_value = 0

        # Iterate over successor states and update the Q-value
        for next_state, prob in successors:
            # Update Q-value using the Bellman equation
            q_value += prob * (self.mdp.getReward(state, action, next_state)
                            + self.discount * self.getValue(next_state))

        # Return the computed Q-value
        return q_value


    def computeActionFromValues(self, state):
        """
        The policy is the best action in the given state
        according to the values currently stored in self.values.

        You may break ties any way you see fit. Note that if
        there are no legal actions, which is the case at the
        terminal state, you should return None.
        """
        # Get the list of possible actions in the given state
        actions = self.mdp.getPossibleActions(state)

        # If there are no legal actions, return None
        if not actions:
            return None

        # Return the action with the maximum Q-value, breaking ties arbitrarily
        return max(actions, key=lambda x: self.computeQValueFromValues(state, x))


    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
