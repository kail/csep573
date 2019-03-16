"""
UW, CSEP 573, Win19
"""
from pomdp import POMDP
from offlineSolver import OfflineSolver
import numpy as np


class QMDP(OfflineSolver):
    def __init__(self, pomdp, precision = .001):
        super(QMDP, self).__init__(pomdp, precision)
        """
        ****Your code
        Remember this is an offline solver, so compute the policy here
        """
        self.V_MDP = [0 for state in self.pomdp.states]
        self.V_QMDP = None
        self.mdpValueIteration()
        print('done')
        
    def mdpValueIteration(self):
        convergence = False
        while not convergence:
            Vk_update = self.mdpValueIterationSingle()
            convergence = all(abs(Vk_update[i] - self.V_MDP[i]) < self.precision for i in range(len(self.V_MDP)))
            self.V_MDP = Vk_update
            
    def mdpValueIterationSingle(self):
        Vk_update = []
        for si, state in enumerate(self.pomdp.states):
            Qsa = []
            for ai, action in enumerate(self.pomdp.actions):
                total = 0
                for nsi, nextState in enumerate(self.pomdp.states):
                    T = self._transitionFn(si, ai, nsi)
                    R = self._rewardFn(si, ai, nsi)
                    Vk = self.V_MDP[nsi]
                    total += T * (R + (self.pomdp.discount * Vk))
                Qsa.append(total)
            
            # Vk_i+1 calc
            Vk_update.append(max(Qsa))
        return Vk_update
            
    def _transitionFn(self, stateIndex, actionIndex, nextStateIndex):
        return self.pomdp.T[actionIndex][stateIndex][nextStateIndex]
    
    def _rewardFn(self, stateIndex, actionIndex, nextStateIndex):
        # TODO: can "observation" be omitted? Summed? Multiplied by prior?
        return self.pomdp.R[actionIndex][stateIndex][nextStateIndex][0]
    
    def chooseAction(self, cur_belief):
        """
        ***Your code
        """
        cur_belief = cur_belief.flatten()
        _ = self.getValue(cur_belief)
        max_v, max_i = max((val, i) for i, val in enumerate(self.V_QMDP))
        return max_i
    
    def getValue(self, belief):
        """
        ***Your code
        """
        V_QMDP =[]
        for ai, action in enumerate(self.pomdp.actions):
            val = self._qmdpBeliefFn(belief, ai)
            V_QMDP.append(val)
        
        self.V_QMDP = V_QMDP
        return max(V_QMDP)
    
    def _qmdpBeliefFn(self, belief, actionIndex):
        total = 0
        for si, state in enumerate(self.pomdp.states):
            qmdp_val = self._qmdpFn(si, actionIndex)
            total += qmdp_val * belief[si]
        return total
            
    def _qmdpFn(self, stateIndex, actionIndex):
        total = 0
        for nsi, nextState in enumerate(self.pomdp.states):
            T = self._transitionFn(stateIndex, actionIndex, nsi)
            total += T * self.V_MDP[nsi]
        R = self._qmdpRewardFn(stateIndex, actionIndex)
        return R + (self.pomdp.discount * total)
    
    def _qmdpRewardFn(self, stateIndex, actionIndex):
        # TODO: should this be a sum?
        return self.pomdp.R[actionIndex][stateIndex][0][0]
    

    """
    ***Your code
    Add any function, data structure, etc that you want
    """
    

class MinMDP(OfflineSolver):
    
    def __init__(self, pomdp, precision = .001):
        super(MinMDP, self).__init__(pomdp, precision)
        """
        ***Your code 
        Remember this is an offline solver, so compute the policy here
        """
        self.V_WORST = None
        self.discount_sum = 1 / (1 - self.pomdp.discount)
    
    def getValue(self, cur_belief):
        """
        ***Your code
        """
        V_WORST = []
        for ai in range(len(self.pomdp.actions)):
            total = 0
            for si in range(len(self.pomdp.states)):
                R = self._valRewardFn(si, ai)
                total += (cur_belief[si] * R)
            
            total += self.pomdp.discount * (self.discount_sum * self._rMinFn())
            V_WORST.append(total)
        self.V_WORST = V_WORST
        return max(V_WORST)

    def chooseAction(self, cur_belief):
        """
        ***Your code
        """  
        cur_belief = cur_belief.flatten()
        _ = self.getValue(cur_belief)
        max_v, max_i = max((val, i) for i, val in enumerate(self.V_WORST))
        return max_i
    
    """
    ***Your code
    Add any function, data structure, etc that you want
    """
    def _valRewardFn(self, stateIndex, actionIndex):
        # TODO: should this be a sum?
        return self.pomdp.R[actionIndex][stateIndex][0][0]
    
    def _rMinFn(self):
        return np.amin(self.pomdp.R)
