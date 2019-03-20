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
        self.V_MDP = np.zeros(len(self.pomdp.states))
        self.V_QMDP = np.zeros((len(self.pomdp.states), len(self.pomdp.actions)))
        
        # Create a const reward array
        self.rewardConst = np.sum(np.multiply(pomdp.R[:,:,:,0], pomdp.T), 2)
        self.mdpValueIteration()
        print('done')
        
    def mdpValueIteration(self):
        convergence = False
        while not convergence:
            Vk_update = self.mdpValueIterationSingle()
            convergence = all(abs(Vk_update[i] - self.V_MDP[i]) < self.precision for i in range(len(self.pomdp.states)))
            self.V_MDP = Vk_update
            
    def mdpValueIterationSingle(self):
        # Changing iterative approach to numpy
        # Vk_update = []
        # for si, state in enumerate(self.pomdp.states):
        #     Qsa = []
        #     for ai, action in enumerate(self.pomdp.actions):
        #         total = 0
        #         for nsi, nextState in enumerate(self.pomdp.states):
        #             T = self._transitionFn(si, ai, nsi)
        #             R = self._rewardFn(si, ai, nsi)
        #             Vk = self.V_MDP[nsi]
        #             total += T * (R + (self.pomdp.discount * Vk))
        #         Qsa.append(total)
        #
        #     # Vk_i+1 calc
        #     Vk_update.append(max(Qsa))
        # return Vk_update
        TQ = np.dot(self.pomdp.T, self.V_MDP) * self.pomdp.discount
        TR = np.add(TQ, self.rewardConst)
        Qsa = np.swapaxes(TR, 0, 1)
        
        Vk_update = np.max(Qsa, axis=1)
        self.V_QMDP = Qsa
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
        # Iterative approach
        # cur_belief = cur_belief.flatten()
        # _ = self.getValue(cur_belief)
        # max_v, max_i = max((val, i) for i, val in enumerate(self.V_QMDP))
        # return max_i
        
        # Numpy approach
        q = np.dot(cur_belief, self.V_QMDP)
        return np.argmax(q)
    
    def getValue(self, belief):
        """
        ***Your code
        """
        # Iterative approach
        # V_QMDP =[]
        # for ai, action in enumerate(self.pomdp.actions):
        #     val = self._qmdpBeliefFn(belief, ai)
        #     V_QMDP.append(val)
        #
        # self.V_QMDP = V_QMDP
        # return max(V_QMDP)
        
        # Numpy approach
        q = np.dot(belief, self.V_QMDP)
        return np.max(q)
    
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
        self.rewardConst = self._valRewardFn()
    
    def getValue(self, cur_belief):
        """
        ***Your code
        """
        # Iterative approach
        # V_WORST = []
        # for ai in range(len(self.pomdp.actions)):
        #     total = 0
        #     for si in range(len(self.pomdp.states)):
        #         R = self._valRewardFn(si, ai)
        #         total += (cur_belief[si] * R)
        #
        #     total += self.pomdp.discount * (self.discount_sum * self._rMinFn())
        #     V_WORST.append(total)
        # self.V_WORST = V_WORST
        # return max(V_WORST)
        
        # Numpy approach
        r_min = self._rMinFn()
        V_a = np.dot(cur_belief, self.rewardConst)
        V_max = np.max(V_a)
        return V_max + (self.pomdp.discount / (1 - self.pomdp.discount)) * r_min

    def chooseAction(self, cur_belief):
        """
        ***Your code
        """  
        # cur_belief = cur_belief.flatten()
        # _ = self.getValue(cur_belief)
        # max_v, max_i = max((val, i) for i, val in enumerate(self.V_WORST))
        # return max_i
        vals = np.dot(cur_belief, self.rewardConst)
        return np.argmax(vals)
    
    """
    ***Your code
    Add any function, data structure, etc that you want
    """
    def _valRewardFn(self):
        # TODO: should this be a sum?
        # return self.pomdp.R[actionIndex][stateIndex][0][0]
        TR = np.multiply(self.pomdp.R[:,:,:,0], self.pomdp.T)
        sum = np.sum(TR, 2)
        return np.swapaxes(sum, 0, 1)
    
    def _rMinFn(self):
        return np.amin(self.pomdp.R)
