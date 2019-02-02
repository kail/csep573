# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util
from enum import Enum

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """
    def feature_nearest_food(self, successorGameState):
        MULTIPLIER = 2
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        
        min_distance = 999999
        for row_index in range(len(newFood.data)):
            for col_index in range(len(newFood.data[row_index])):
                if newFood.data[row_index][col_index]:
                    distance = util.manhattanDistance(newPos, (row_index, col_index))#abs(newPos[0] - row) + abs(newPos[1] - col)
                    if distance < min_distance:
                        min_distance = distance
        
        return MULTIPLIER / min_distance
    
    def feature_sum_food_distances(self, successorGameState):
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        
        distance_reciprocal_sum = 0
        total_food = 0
        for row_index in range(len(newFood.data)):
            for col_index in range(len(newFood.data[row_index])):
                if newFood.data[row_index][col_index]:
                    distance = util.manhattanDistance(newPos, (row_index, col_index))#abs(newPos[0] - row) + abs(newPos[1] - col)
                    distance_reciprocal_sum += 1/distance
                    total_food += 1
        
        return distance_reciprocal_sum# / total_food if total_food else 0
    
    def feature_nearby_ghost(self, successorGameState):
        MULTIPLIER = -20
        newPos = successorGameState.getPacmanPosition()
        newGhostStates = successorGameState.getGhostStates()
        
        distance_sum = 0
        for ghostState in newGhostStates:
            # Only include ghosts if scaredTime < distance and the ghost is close
            distance = util.manhattanDistance(newPos, ghostState.getPosition())
            if ghostState.scaredTimer < distance and distance < 8:
                distance_sum += distance
                
        return MULTIPLIER * len(newGhostStates) / distance_sum if distance_sum else 0
    
    # def feature_pos_has_food(self, successorGameState):
    #     newPos = successorGameState.getPacmanPosition()
    #     newFood = successorGameState.getFood()
    #
    #     if newFood.data[newPos[0]][newPos[1]]:
    #         return 1
    #     return -1
    
    def feature_score_delta(self, successorGameState):
        return successorGameState.data.scoreChange
    
    def feature_state_visited(self, successorGameState):
        if successorGameState in successorGameState.explored:
            return -5
        return 0

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        feature_funcs = [
            self.feature_nearest_food,
            self.feature_sum_food_distances,
            self.feature_nearby_ghost,
            self.feature_score_delta,
            self.feature_state_visited,
        ]
        score = 0
        for feature in feature_funcs:
            score += feature(successorGameState)
            
        if action == 'Stop':
            score -= 1
        
        return score

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxNode:
    class Layer(Enum):
        MIN = 0
        MAX = 1
    
    def __init__(self, layer, game_state, action=None):
        self.edges = []
        self.layer = layer
        self.game_state = game_state
        self.action = action
        self.edge_results = {}
        
    def evaluate(self, function, return_action=False):
        if not self.edges:
            return function(self.game_state)
        
        if self.layer == MinimaxNode.Layer.MIN:
            return min(edge.evaluate(function=function) for edge in self.edges)
        else:
            edge_values = [edge.evaluate(function=function) for edge in self.edges]
            max_edge = float('-inf')
            max_edge_index = None
            for i in range(len(edge_values)):
                if edge_values[i] > max_edge:
                    max_edge = edge_values[i]
                    max_edge_index = i
            
            if return_action:
                return self.edges[max_edge_index].action
            return max_edge
        
    def evaluate_ab(self, function, alpha, beta, return_action=False):
        if not self.edges:
            return function(self.game_state)
        
        if self.layer == MinimaxNode.Layer.MIN:
            val = float('inf')
            for edge in self.edges:
                edge_result = edge.evaluate_ab(function, alpha, beta)
                val = min(val, edge_result)
                if val < alpha:
                    return val
                beta = min(beta, val)
            return val
        else:
            val = float('-inf')
            for edge in self.edges:
                edge_result = edge.evaluate_ab(function, alpha, beta)
                if return_action:
                    self.edge_results[edge] = edge_result
                val = max(val, edge_result)
                if val > beta:
                    return val
                alpha = max(val, alpha)
            if not return_action:
                return val
                
            # Run this only from the root max node (when return_action == True)
            max_result = float('-inf')
            max_edge = None
            for edge in self.edge_results:
                if self.edge_results[edge] > max_result:
                    max_result = self.edge_results[edge]
                    max_edge = edge
            return max_edge.action
            

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """
    
    def __init__(self, *arg, **kwargs):
        super().__init__(*arg, **kwargs)
        self.next_action = None
        
    def evaluate(self, depth, agent_index, game_state):
        # Update the depth if we are at a new level
        if agent_index == game_state.getNumAgents():
            agent_index = 0
            depth += 1
        
        # 0-indexed depth
        if depth == self.depth:
            return self.evaluationFunction(game_state)
        
        if agent_index != 0:
            val = float('inf')
            legal_actions = game_state.getLegalActions(agent_index)
            if not legal_actions:
                return self.evaluationFunction(game_state)
            for action in legal_actions:
                successor_state = game_state.generateSuccessor(agent_index, action)
                edge_result = self.evaluate(depth, agent_index + 1, successor_state)
                val = min(val, edge_result)
            return val
        else:
            set_action = (depth == 0 and agent_index == 0)
            
            val = float('-inf')
            legal_actions = game_state.getLegalActions(agent_index)
            if not legal_actions:
                return self.evaluationFunction(game_state)
            for action in legal_actions:
                successor_state = game_state.generateSuccessor(agent_index, action)
                edge_result = self.evaluate(depth, agent_index + 1, successor_state)
                
                # Sets the next action to be taken
                if set_action and edge_result > val:
                    self.next_action = action
                val = max(val, edge_result)
            return val

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        
        # --- Working implementation ---
        # num_agents = gameState.getNumAgents()
        # root_node = MinimaxNode(layer=MinimaxNode.Layer.MAX, game_state=gameState)
        # leaves = [root_node]
        # for depth in range(self.depth):
        #     for agent_index in range(num_agents):
        #         layer = MinimaxNode.Layer.MAX if agent_index == num_agents - 1 else MinimaxNode.Layer.MIN
        #         new_leaves = []
        #         for leaf in leaves:
        #             game_state = leaf.game_state
        #             legal_actions = game_state.getLegalActions(agent_index)
        #             action_successors = [(action, game_state.generateSuccessor(agent_index, action)) for action in legal_actions]
        #
        #             # Create new node
        #             for action, successor in action_successors:
        #                 new_leaf = MinimaxNode(layer=layer, game_state=successor, action=action)
        #                 leaf.edges.append(new_leaf)
        #                 new_leaves.append(new_leaf)
        #
        #         # Reset the leaves to look at
        #         leaves = new_leaves
        #
        # # Evaluate the minimax func
        # return root_node.evaluate(self.evaluationFunction, return_action=True)
        self.evaluate(0, 0, gameState)
        return self.next_action

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """
    def __init__(self, *arg, **kwargs):
        super().__init__(*arg, **kwargs)
        self.next_action = None
        
    def evaluate(self, depth, agent_index, game_state, alpha, beta):
        # Update the depth if we are at a new level
        if agent_index == game_state.getNumAgents():
            agent_index = 0
            depth += 1
        
        # 0-indexed depth
        if depth == self.depth:
            return self.evaluationFunction(game_state)
        
        if agent_index != 0:
            val = float('inf')
            legal_actions = game_state.getLegalActions(agent_index)
            if not legal_actions:
                return self.evaluationFunction(game_state)
            for action in legal_actions:
                successor_state = game_state.generateSuccessor(agent_index, action)
                edge_result = self.evaluate(depth, agent_index + 1, successor_state, alpha, beta)
                val = min(val, edge_result)
                if val < alpha:
                    return val
                beta = min(beta, val)
            return val
        else:
            set_action = depth == 0 and agent_index == 0
            
            val = float('-inf')
            legal_actions = game_state.getLegalActions(agent_index)
            if not legal_actions:
                return self.evaluationFunction(game_state)
            for action in legal_actions:
                successor_state = game_state.generateSuccessor(agent_index, action)
                edge_result = self.evaluate(depth, agent_index + 1, successor_state, alpha, beta)
                
                # Sets the next action to be taken
                if set_action and edge_result > val:
                    self.next_action = action
                val = max(val, edge_result)
                if val > beta:
                    return val
                alpha = max(val, alpha)
            return val
        
    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        self.evaluate(0, 0, gameState, alpha=float('-inf'), beta=float('inf'))
        return self.next_action

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
