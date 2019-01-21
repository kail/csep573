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
    
    visited_set = set()
    
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
            
        if state in visited_set:
            continue
        visited_set.add(state)
            
        # Get all potential successors
        successors = problem.getSuccessors(state)
        for successor in successors:
            successor_state = successor[0]
            step_cost = successor[2]
            
            if successor_state in visited_set or successor_state in parent_graph:
                continue
            
            # if successor_state in parent_graph:
            #     # This node has been visited. Skip
            #     continue
                
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
    import copy
    
    data_structure = util.Stack()
    
    # This will map the visited nodes to a tuple (parent node loc, successor)
    parent_graph = dict()
    
    # Used to report the result at the end of the search
    result_queue = util.Queue()
    
    visited_set = set()
    
    # Initialize the stack or queue with the starting state
    start_state = problem.getStartState()
    visited_set.add(start_state)
    parent_graph[start_state] = None
    data_structure.push((start_state, None, 0))
        
    # Iterate while data_structure has something in it
    end_successor = None
    while not data_structure.isEmpty():
        state, successor_val, cost_so_far = data_structure.pop()

        # End iteration if we have reached the goal
        if problem.isGoalState(state):
            end_successor = successor_val
            break
            
        visited_set.add(state)
        
        # Get all potential successors
        successors = problem.getSuccessors(state)
        for successor in successors:
            successor_state = successor[0]
            
            if successor_state in visited_set:
                continue

            # Save the successor to graph, and queue
            parent_graph[successor_state] = (state, successor_val)
            
            # Add the next nodes to the structure
            data_structure.push((successor_state, successor, cost_so_far))
                
    # Reconstruct the path we took to get here
    parent_successor = end_successor
    while parent_successor:
        result_queue.push(parent_successor)
        parent_successor = parent_graph[parent_successor[0]][1]
        
    steps = [step[1] for step in result_queue.list]
    return copy.copy(steps)


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    #return genericSearch(problem, util.Queue())
    
    import copy
    
    data_structure = util.Queue()
    
    # This will map the visited nodes to a tuple (parent node loc, successor)
    parent_graph = dict()
    
    # Used to report the result at the end of the search
    result_queue = util.Queue()
    
    visited_set = set()
    
    # Initialize the stack or queue with the starting state
    start_state = problem.getStartState()
    
    parent_graph[start_state] = None
    data_structure.push((start_state, None, 0))
        
    # Iterate while data_structure has something in it
    end_successor = None
    while not data_structure.isEmpty():
        state, successor_val, cost_so_far = data_structure.pop()

        # End iteration if we have reached the goal
        if problem.isGoalState(state):
            end_successor = successor_val
            break
            
        if state in visited_set:
            continue
        visited_set.add(state)
        
        # Get all potential successors
        successors = problem.getSuccessors(state)
        for successor in successors:
            successor_state = successor[0]
            
            if successor_state in visited_set or successor_state in parent_graph:
                continue

            # Save the successor to graph, and queue
            parent_graph[successor_state] = (state, successor_val)
            
            # Add the next nodes to the structure
            data_structure.push((successor_state, successor, cost_so_far))
                
    # Reconstruct the path we took to get here
    parent_successor = end_successor
    while parent_successor:
        result_queue.push(parent_successor)
        parent_successor = parent_graph[parent_successor[0]][1]
        
    steps = [step[1] for step in result_queue.list]
    return copy.copy(steps)


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def _aStarSearch(problem, heuristic=nullHeuristic):
    # Map of parent nodes
    came_from = {}
    
    def reconstruct_path(current):
        total_path = []
        current = (current, None)
        while current[0] in came_from:
            current = came_from[current[0]]
            total_path.append(current[1])
        total_path.reverse()
        return total_path
    
    # Previously seen nodes
    closed_set = set()

    start_state = problem.getStartState()
    
    # Initially, only the start node is known
    open_queue = util.PriorityQueue()
    open_queue.push(start_state, 0)
    
    # The set of currently discovered nodes that are not evaluated yet.
    open_set = {start_state}
    
    # For each node, the cost of getting from the start node to that node.
    global_score = {}
    
    # The cost of going from start to start is zero.
    global_score[start_state] = 0
    
    # f_score = heuristic + global_score
    f_score = {}
    f_score[start_state] = heuristic(start_state, problem)
    
    while not open_queue.isEmpty():
        current_state = open_queue.pop()
        if problem.isGoalState(current_state):
            return reconstruct_path(current_state)

        # This could have been removed previously, if 2 of the same states were on the queue with different priorities
        if current_state in open_set:
            open_set.remove(current_state)
        closed_set.add(current_state)

        neighbors = problem.getSuccessors(current_state)
        for neighbor in neighbors:
            if neighbor[0] == 'G':
                print('debug')
            if neighbor[0] in closed_set:
                continue  # Ignore the neighbor which is already evaluated.

            tentative_gScore = global_score[current_state] + neighbor[2]
            added_to_set = False
            if neighbor[0] not in open_set:
                added_to_set = True
                open_set.add(neighbor[0])
            elif tentative_gScore >= global_score[neighbor[0]]:
                continue

            came_from[neighbor[0]] = (current_state, neighbor[1])
            global_score[neighbor[0]] = tentative_gScore
            f_score[neighbor[0]] = global_score[neighbor[0]] + heuristic(neighbor[0], problem)
            
            
            open_queue.push(neighbor[0], f_score[neighbor[0]])


def uniformCostSearch(problem):
    """Search the shallowest nodes in the search tree first, but incorporate cost"""
    #return genericSearch(problem, util.PriorityQueue())
    return _aStarSearch(problem)


def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    # NOTE: AStar search was implemented outside of generic search because it wasn't working as expected
    return _aStarSearch(problem, heuristic)


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
