# rtdpAgents.py
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
# Pieter Abbeel (pabbeel@cs.berkeley.edu). The RTDP question was added by
# Gagan Bansal (bansalg@cs.washington.edu) and Dan Weld (weld@cs.washington.edu).


import random
import mdp, util
import math

from learningAgents import ValueEstimationAgent

class RTDPAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A RTDPAgent takes a Markov decision process
        (see mdp.py) on initialization and runs rtdp 
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, max_iters=100):
        """
          Your value rtdp agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
              mdp.getStartState()

          Other useful functions:
              weighted_choice(choices)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = {}  # note, we use a normal python dictionary for RTDPAgent.

        # Write rtdp code here
        self.start_state = mdp.getStartState()
        self.states = mdp.getStates()
        self.use_stack = True
        self.stack = []
        
        for _ in range(self.iterations):
            current_state = self.start_state
            iter = 0
            while not self.mdp.isTerminal(current_state) and iter < max_iters:
                iter += 1
                # Pick the best action and update hash
                action, utility = self.greedyAction(current_state)
                
                if not self.use_stack:
                    self.values[current_state] = utility
                else:
                    self.stack.append((current_state, utility))
                
                # Stochastically simulate next state
                current_state = self.pickNextState(current_state, action)
            
            if self.use_stack and self.stack:
                for i in range(len(self.stack)):
                    back_index = len(self.stack) - i - 1
                    item = self.stack[back_index]
                    self.values[item[0]] = item[1]
                # Reset
                self.stack = []
        
    def greedyAction(self, state):
        actions = self.mdp.getPossibleActions(state)
        
        actionCounter = util.Counter()
        for action in actions:
            actionCounter[action] = self.getQValue(state, action)
        
        pi = actionCounter.argMax()
        return pi, actionCounter[pi]
    
    def pickNextState(self, state, action):
        """
        Return the next stochastically simulated state.
        """
        stateProbabilities = [stateProb for stateProb in self.mdp.getTransitionStatesAndProbs(state, action)]
        return weighted_choice(stateProbabilities)

    def updateValue(self, state, action):
        """
        Update the value of given state.
        """
        self.values[state] = self.getQValue(state, action)

    def getHeuristicValue(self, state):
        """
        Return the heuristic value of state.
        """
        
        # Should be upper bound on value
        if self.mdp.isTerminal(state):
            return 0
        potential_reward = self.mdp.getReward(state, None, None)
        if potential_reward:
            return potential_reward
        
        return math.pow(self.discount, util.manhattanDistance(self.mdp.getGoalState(), state)) * self.mdp.getGoalReward()
        #return self.mdp.getGoalReward()
        
        # goalState = self.mdp.getGoalState()
        # if goalState == state or self.mdp.isTerminal(state):
        #     return self.mdp.getGoalReward() * 2
        #
        # return self.mdp.getGoalReward() + (self.mdp.getGoalReward() / (util.manhattanDistance(goalState, state) + 1))

    def getValue(self, state):
        """
          Return the current stored value of the state.
          If the state has not been seen yet then return it heuristic value.

          Note the difference between this and the similar method in valueIterationAgents
        """
        value = None
        if state not in self.values:
            value = self.getHeuristicValue(state)
            self.values[state] = value
        return self.values[state]

    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        transitionStateProbs = self.mdp.getTransitionStatesAndProbs(state, action)
        totalQ = 0
        
        q_vals = []
        
        for transitionState, transitionProb in transitionStateProbs:
            reward = self.mdp.getReward(state, action, transitionState)
            utility = self.getValue(transitionState)
            q_val = transitionProb * ((self.discount * utility) + reward)
            #q_val = transitionProb * ((self.discount * utility))
            #totalQ += q_val
            q_vals.append(q_val)
        
        return sum(q_vals)
        #return totalQ

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        possibleActions = self.mdp.getPossibleActions(state)
        if not possibleActions:
            return None
        
        counter = util.Counter()
        for action in possibleActions:
            counter[action] = self.getQValue(state, action)
        
        return counter.argMax()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)


def weighted_choice(choices):
    """
    Return a random element from list of the form: [(choice, weight), ....]
    Credits: http://stackoverflow.com/questions/3679694
    """
    total = sum(w for c, w in choices)
    r = random.uniform(0, total)
    upto = 0
    for c, w in choices:
       if upto + w >= r:
          return c
       upto += w
    assert False, "Shouldn't get here"