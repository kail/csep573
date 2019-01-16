# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]


def genericSearch(problem, data_structure, heuristic = None):
    import copy
    
    # This will map the visited nodes to a tuple (parent node loc, successor)
    parent_graph = dict()
    
    # Used to report the result at the end of the search
    result_queue = util.Queue()
    
    # Initialize the stack or queue with the starting state
    start_state = problem.getStartState()
    parent_graph[start_state] = None
    if isinstance(data_structure, util.PriorityQueue):
        data_structure.push((start_state, None, 0), priority=0)
    else:
        data_structure.push((start_state, None, 0))
        
    # Iterate while data_structure has something in it
    end_successor = None
    while not data_structure.isEmpty():
        state, successor_val, cost_so_far = data_structure.pop()

        # End iteration if we have reached the goal
        if problem.isGoalState(state):
            end_successor = successor_val
            break
            
        # Get all potential successors
        successors = problem.getSuccessors(state)
        for successor in successors:
            successor_state = successor[0]
            step_cost = successor[2]
            
            if successor_state in parent_graph:
                # This node has been visited. Skip
                continue
                
            # Update the total cost
            cost_so_far += step_cost
            
            # For uniform search, priority = step_cost at each step
            priority = step_cost
            
            # If we have a heuristic, priority needs to have total cost and s
            if heuristic:
                heuristicCost = heuristic(successor_state, problem)
                priority = heuristicCost + cost_so_far
            
            # Save the successor to graph, and queue
            parent_graph[successor_state] = (state, successor_val)
            
            # Add the next nodes to the structure
            if isinstance(data_structure, util.PriorityQueue):
                data_structure.push((successor_state, successor, cost_so_far), priority=priority)
            else:
                data_structure.push((successor_state, successor, cost_so_far))
                
    # Reconstruct the path we took to get here
    parent_successor = end_successor
    while parent_successor:
        result_queue.push(parent_successor)
        parent_successor = parent_graph[parent_successor[0]][1]
        
    steps = [step[1] for step in result_queue.list]
    return copy.copy(steps)


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.
    """
    return genericSearch(problem, util.Stack())


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    return genericSearch(problem, util.Queue())


def uniformCostSearch(problem):
    """Search the shallowest nodes in the search tree first, but incorporate cost"""
    return genericSearch(problem, util.PriorityQueue())


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def _aStarSearch(problem, heuristic=nullHeuristic):
    # // The set of nodes already evaluated
    # closedSet := {}
    closedSet = set()
    
    #
    # // The set of currently discovered nodes that are not evaluated yet.
    # // Initially, only the start node is known.
    # openSet := {start}
    start_state = problem.getStartState()
    openQueue = util.PriorityQueue()
    openQueue.push(start_state, 0)
    openSet = {start_state}
    
    #
    # // For each node, which node it can most efficiently be reached from.
    # // If a node can be reached from many nodes, cameFrom will eventually contain the
    # // most efficient previous step.
    # cameFrom := an empty map
    cameFrom = {}
    
    #
    # // For each node, the cost of getting from the start node to that node.
    # gScore := map with default value of Infinity
    gScore = {}
    
    #
    # // The cost of going from start to start is zero.
    # gScore[start] := 0
    gScore[start_state] = 0
    
    #
    # // For each node, the total cost of getting from the start node to the goal
    # // by passing by that node. That value is partly known, partly heuristic.
    # fScore := map with default value of Infinity
    fScore = {}
    
    # function reconstruct_path(cameFrom, current)
    def reconstruct_path(current):
        # total_path := {current}
        total_path = []
        current = (current, None)
        # while current in cameFrom.Keys:
        while current[0] in cameFrom:
    #       current := cameFrom[current]
            current = cameFrom[current[0]]
    #       total_path.append(current)
            total_path.append(current[1])
        total_path.reverse()
        return total_path
    
    #
    # // For the first node, that value is completely heuristic.
    # fScore[start] := heuristic_cost_estimate(start, goal)
    fScore[start_state] = heuristic(start_state, problem)
    
    #
    # while openSet is not empty
    while not openQueue.isEmpty():
    #   current := the node in openSet having the lowest fScore[] value
        current_state = openQueue.pop()
    #   if current = goal
        if problem.isGoalState(current_state):
            return reconstruct_path(current_state)
    #
    #   openSet.Remove(current) -- done with pop
        openSet.remove(current_state)
    
    #   closedSet.Add(current)
        closedSet.add(current_state)
    #
    
    #   for each neighbor of current
        neighbors = problem.getSuccessors(current_state)
        for neighbor in neighbors:

            if neighbor[0] in closedSet:
                continue  # Ignore the neighbor which is already evaluated.
    #
    #       // The distance from start to a neighbor
    #       tentative_gScore := gScore[current] + dist_between(current, neighbor)
            tentative_gScore = gScore[current_state] + neighbor[2]
    #
    #       if neighbor not in openSet	// Discover a new node
            added_to_set = False
            if neighbor[0] not in openSet:
                added_to_set = True
                openSet.add(neighbor[0])
            elif tentative_gScore >= gScore[neighbor[0]]: # TODO: watch this var!!
                continue
    #
    #       // This path is the best until now. Record it!
    #       cameFrom[neighbor] := current
            cameFrom[neighbor[0]] = (current_state, neighbor[1])
    #       gScore[neighbor] := tentative_gScore
            gScore[neighbor[0]] = tentative_gScore
    #       fScore[neighbor] := gScore[neighbor] + heuristic_cost_estimate(neighbor, goal)
            fScore[neighbor[0]] = gScore[neighbor[0]] + heuristic(neighbor[0], problem)
            
            if added_to_set:
                openQueue.push(neighbor[0], fScore[neighbor[0]])

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    #return genericSearch(problem, util.PriorityQueue(), heuristic)
    return _aStarSearch(problem, heuristic)


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
