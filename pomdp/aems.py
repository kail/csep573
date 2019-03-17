#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UW, CSEP 573, Win19
"""

from pomdp import POMDP
from onlineSolver import OnlineSolver
from typing import List, Optional, Set
import numpy as np


class Node:
    def __init__(self):
        self.U: float = float('inf')
        self.L: float = float('-inf')
        
class ActionNode(Node):
    def __init__(self, ai: int, parent: Node, reward: float):
        super(Node, self).__init__()
        if not isinstance(parent, BeliefNode):
            raise Exception("Needs to be a belief node!")
        
        self.parent: BeliefNode = parent        # Belief node that points to this action node
        self.ai = ai                            # Action index
        self.reward: float = reward             # Arc from belief node = R(b, a)
        self.children: List[BeliefNode] = []    # Children belief nodes
        
    
class BeliefNode(Node):
    def __init__(
        self,
        belief,
        parent: Optional[Node],
        depth: int,
        gamma_d: float,
        oi: int,
        obs_prob: float,
        obs_prob_cumulative: float,
    ):
        super(Node, self).__init__()
        if parent and not isinstance(parent, ActionNode):
            raise Exception("Needs to be an action node!")
        
        self.parent: ActionNode = parent                # action node pointing to this belief node
        self.belief = belief                            # numpy array
        self.children: List[ActionNode] = []            # action nodes that this points to
        self.depth = depth                              # depth
        self.gamma_d = gamma_d                          # discount^depth
        self.oi = oi                                    # observation index
        self.obs_prob = obs_prob                        # probability of seeing the observation P(z | b, a) (arc from AND node)
        self.obs_prob_cumulative = obs_prob_cumulative  # Pi[P(z|b,a) * P(a|b)] from i=0 to i=depth
        self.chosen_action_index = None
        
        
class AEMS2(OnlineSolver):
    def __init__(self, pomdp, lb_solver, ub_solver, precision = .001, action_selection_time = .1):
        super(AEMS2, self).__init__(pomdp, precision, action_selection_time)
        self.lb_solver = lb_solver
        self.ub_solver = ub_solver
        """
        *****Your code
        You can add any attribute you want
        """
        
        # Collections of all belief and action nodes
        self.belief_nodes: Set[BeliefNode] = set()
        self.action_nodes: Set[ActionNode] = set()
        
        # Initial belief comes from the prior
        initial_belief = np.copy(self.pomdp.prior)
        initial_L = self.lb_solver.getValue(initial_belief)
        initial_U = self.ub_solver.getValue(initial_belief)
        
        self.root: BeliefNode = BeliefNode(
            belief=initial_belief,
            parent=None,
            depth=0,
            gamma_d=1,
            oi=0,
            obs_prob=1.0,
            obs_prob_cumulative=1.0
        )
        self.root.L = initial_L
        self.root.U = initial_U
        self.belief_nodes.add(self.root)
    
    #
    # Choose
    #
    def is_fringe_node(self, belief_node: BeliefNode) -> bool:
        if not belief_node.children:
            return True
        return False
    
    def get_all_fringe_nodes(self) -> List[BeliefNode]:
        fringe_nodes: List[BeliefNode] = []
        for bn in self.belief_nodes:
            fringe_nodes.append(bn)
        return fringe_nodes
    
    def select_best_fringe_node(self) -> BeliefNode:
        # If root is a fringe node, then return it
        if self.is_fringe_node(self.root):
            return self.root
        
        # Otherwise, select the next actions
        fringe_nodes = self.get_all_fringe_nodes()
        max = float('-inf')
        max_i = -1
        
        # b* ← arg maxb∈FRINGE(G) E(b)
        for i, bn in enumerate(fringe_nodes):
            e = self.E(bn)
            if e > max:
                max = e
                max_i = i
        return fringe_nodes[max_i]
    
    # E(b) = gamma^d * P(b) * e_hat(b)
    def E(self, bn: BeliefNode) -> float:
        return bn.gamma_d * self.e_hat(bn) * self.P(bn)
        
    def e_hat(self, bn: BeliefNode) -> float:
        return bn.U - bn.L
    
    def P(self, bn: BeliefNode):
        return bn.obs_prob_cumulative

    #
    # Expand
    #
    def expand(self, bn: BeliefNode):
        L_a_max = float('-inf')
        U_a_max = float('-inf')
        
        for ai, action in enumerate(self.pomdp.actions):
            L_a = U_a = reward = self.R_b_a(bn, ai)
            
            # Create new action node
            new_an = ActionNode(ai=ai, parent=bn, reward=reward)
            
            for oi, obs in enumerate(self.pomdp.observations):
                prob_arc_val = self.P_o_b_a(bn, ai, oi)
                
                # Calculate new belief
                # TODO!! Should oi be from the current observation or from the past (bn.oi)?
                new_belief = self.NewBelief(bn=bn, ai=ai, oi=oi)
                
                # Use heuristics to get U and L
                L = self.lb_solver.getValue(new_belief)
                U = self.ub_solver.getValue(new_belief)
                
                # TODO: Is this correct?
                # Equation 2 - set the action node L and U
                L_a += self.pomdp.discount * prob_arc_val * L
                U_a += self.pomdp.discount * prob_arc_val * U
                
                # TODO: Is this correct?
                # P(b^d) = Pi[P(o|b,a)*P(a|b)
                obs_prob_cumulative = bn.obs_prob_cumulative * bn.obs_prob
                
                # TODO: Create a new belief node and append to action node
                new_bn = BeliefNode(
                    belief=new_belief,
                    parent=new_an,
                    depth=bn.depth+1,
                    gamma_d=bn.gamma_d * self.pomdp.discount,
                    oi=oi,
                    obs_prob=prob_arc_val,
                    obs_prob_cumulative=obs_prob_cumulative,
                )
                new_bn.L = L
                new_bn.U = U
                
                # Add the new BN to the new AN and to the belief node set (for searching)
                new_an.children.append(new_bn)
                self.belief_nodes.add(new_bn)
            
            # Get the max vals to update the current bn
            if L_a > L_a_max:
                L_a_max = L_a
            if U_a > U_a_max:
                U_a_max = U_a
                
            # Configure the action node and append to chosen bn (created by eq 2)
            new_an.L = L_a
            new_an.U = U_a
            bn.children.append(new_an)
        
        # TODO: Does this actually need to happen?
        bn.U = U_a_max
        bn.L = L_a_max
        
    def R_b_a(self, bn: BeliefNode, ai: int) -> float:
        return 0.0
    
    def P_o_b_a(self, bn: BeliefNode, ai: int, oi: int) -> float:
        return 0.0
    
    # This calculates b'(s') using EQ 1 from the paper
    def NewBelief(self, bn: BeliefNode, ai: int, oi: int):
        b_prime = np.zeros(len(self.pomdp.states))
        
        for s_prime in range(len(self.pomdp.states)):
            O = self.pomdp.O[ai, s_prime, oi]
            s_sum = sum((self.pomdp.T[ai, si, s_prime]*bn.belief[si]) for si in range(len(self.pomdp.states)))
            b_prime[s_prime] = O * s_sum
        
        # Apply normalization
        nf = np.sum(b_prime)
        b_prime = np.divide(b_prime, nf)
        return b_prime
    
    #
    # Backtrack
    #
    def backtrack(self, bn: BeliefNode, L_old: float, U_old: float):
        while bn != self.root:
            an = bn.parent
            an.L += self.pomdp.discount * bn.obs_prob * (bn.L - L_old)
            an.U += self.pomdp.discount * bn.obs_prob * (bn.U - U_old)
            
            # TODO: is there really a typing error here?
            bn = an.parent
            L_old, U_old = self.update_belief_node(bn)
    
    def update_belief_node(self, bn: BeliefNode):
        L_old, U_old = bn.L, bn.U
        max_ai = -1
        U_max = L_max = float('-inf')
        
        for ai, an in enumerate(bn.children):
            if an.U > U_max:
                U_max = an.U
                max_ai = ai
            if an.L > L_max:
                L_max = an.L
                
        bn.L = L_max
        bn.U = U_max
        bn.chosen_action_index = max_ai
        return L_old, U_old
    
    def expandOneNode(self):
        """
        *****Your code
        """
        # Choose
        best_fringe_node = self.select_best_fringe_node()
        L_old, U_old = best_fringe_node.L, best_fringe_node.U
        
        # Expand
        self.expand(best_fringe_node)
        
        # Backtrack
        self.backtrack(bn=best_fringe_node, L_old=L_old, U_old=U_old)

    def chooseAction(self) -> int:
        """
        *****Your code
        """
        if self.root.chosen_action_index is not None:
            return self.root.chosen_action_index
        
        # This should not execute...
        print('WE SHOULDNT GET HERE')
        max_U = float('-inf')
        max_ai = -1
        for ai, an in enumerate(self.root.children):
            if an.U > max_U:
                max_ai = ai
                max_U = an.U
        
        return max_ai
   
        
    def updateRoot(self, action, observation):
        """
        ***Your code 
        """
        # TODO: May want to throw a bunch of stuff away (plus their children) instead of keeping it in memory
        chosen_an = self.root.children[action]
        throwaway_action_nodes = [an for an in self.root.children if an != chosen_an]
        chosen_bn = chosen_an.children[observation]
        throwaway_belief_nodes = [bn for bn in chosen_an.children if bn != chosen_bn]
        self.root = chosen_bn
        self.root.parent = None
