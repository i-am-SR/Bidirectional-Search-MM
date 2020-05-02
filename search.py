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
#
#Code updated by Sumit Rawat (srawat7@asu.edu)


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util
from util import Queue
import numpy as np

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

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    "*** YOUR CODE HERE ***"
    curr_state = problem.getStartState()
    visited_states = set()	#to store the states already visited
    path = []	
    stack = util.Stack()	#fringe for DFS is a LIFO stack
    stack.push((curr_state, path))
    while not problem.isGoalState(curr_state):	#loop till we reach goal state
    	if curr_state not in visited_states:
    		visited_states.add(curr_state)	#mark state as visited
    		successor_list = problem.getSuccessors(curr_state)	#get successors
    		for (successor_state, next_direction, _) in successor_list:
    			stack.push((successor_state, path + [next_direction]))
    	(curr_state, path) = stack.pop()
    return path
    #util.raiseNotDefined()

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    curr_state = problem.getStartState()
    visited_states = set()	#to store the states already visited
    path = []
    queue = util.Queue()	#fringe for DFS is a FIFO queue
    queue.push((curr_state, path))
    while not problem.isGoalState(curr_state):	#loop till we reach goal state
    	if curr_state not in visited_states:
    		visited_states.add(curr_state)	#mark state as visited
    		successor_list = problem.getSuccessors(curr_state) #get successors
    		for (successor_state, next_direction, _) in successor_list:
    			queue.push((successor_state, path + [next_direction]))
    	(curr_state, path) = queue.pop()
    return path
    #util.raiseNotDefined()

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    curr_state = problem.getStartState()
    visited_states = set()	#to store the states already visited
    path = []
    curr_cost = 0
    pQueue = util.PriorityQueue()	#fringe for UCS is a Priority queue
    pQueue.push((curr_state, path, curr_cost), curr_cost)
    while not problem.isGoalState(curr_state):	#loop till we reach goal state
    	if curr_state not in visited_states:
    		visited_states.add(curr_state)	#mark state as visited
    		successor_list = problem.getSuccessors(curr_state)	#get successors
    		for (successor_state, next_direction, additional_cost) in successor_list:
    			pQueue.push((successor_state, path + [next_direction], curr_cost + additional_cost), curr_cost + additional_cost)
    	(curr_state, path, curr_cost) = pQueue.pop()
    return path
    #util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def nullHeuristic_bi(position, problem, dir = 0, info={}):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided bidirectional SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    curr_state = problem.getStartState()
    visited_states = set()	#to store the states already visited
    path = []
    curr_cost = 0
    pQueue = util.PriorityQueue()	#fringe for A* is a Priority queue
    pQueue.push((curr_state, path, curr_cost), curr_cost + heuristic(curr_state, problem))
    while not problem.isGoalState(curr_state):	#loop till we reach goal state
        if curr_state not in visited_states:
            visited_states.add(curr_state)	#mark state as visited
            successor_list = problem.getSuccessors(curr_state)	#get successors
            for (successor_state, next_direction, additional_cost) in successor_list:
                pQueue.push((successor_state, path + [next_direction], curr_cost + additional_cost), curr_cost + additional_cost + heuristic(successor_state, problem))
        (curr_state, path, curr_cost) = pQueue.pop()
    return path
    #util.raiseNotDefined()


def bidirectionalMMsearch(problem, heuristic=nullHeuristic):
    """Implementation of bidirectional A* search that Meets in the Middle
    A search starts from the start node towards the goal node and another
    starts from teh goal node towards the start node. These searches meet in the middle
    and the cheapest solution is found.
    """
    #start point of the forward search
    curr_state_fwd = problem.getStartState()
    # start point of the backward search
    curr_state_bck = problem.getGoalState()
    # Lists to store the actions followed for the forward and backward searches
    path_fwd = []
    path_bck = []
    # dictonaries for states and their g values in the forward and backward direction
    g_fwd = {curr_state_fwd: 0}
    g_bck = {curr_state_bck: 0}
    # open and closed lists in the forward and backward directions
    open_fwd = [(curr_state_fwd, path_fwd)]
    open_bck = [(curr_state_bck, path_bck)]
    closed_fwd = []
    closed_bck = []
    # U = Cost of the cheapest solution found so far
    U = np.inf

    def search_dir(U, open1, open2, g1, g2, closed, dir):
        "Search in the direction dir"
        n, path = min_p_g(C, open1, g1, dir)
        open1.remove((n, path))
        closed.append((n, path))
        successor_list = problem.getSuccessors(n)
        for (c, next_direction, additional_cost) in successor_list:
            if found(open1, c) or found(closed, c):
                if g1[c] <= g1[n] + additional_cost:
                    continue

                open1 = delete(open1, c)

            g1[c] = g1[n] + additional_cost
            open1.append((c, path + [next_direction]))
            #visited_states.add(c)
            if found(open2, c):
                U = min(U, g1[c] + g2[c])

        return U, open1, closed, g1

    def delete(open1, n):
        """Delete state n from Open list open1"""
        for (c, path) in open1:
            if c == n:
                open1.remove((c, path))
        return open1

    def found(open1, n):
        """Check if the state n is on the Open list open1"""
        for (c, path) in open1:
            if c == n:
                return True
        return False

    def choose_min_n(open1, g, dir):
        """Function to find the minimum values of f and g
        for the states in the open list in the current direction"""
        prmin, prmin_F = np.inf, np.inf
        for (n, path) in open1:
            f = g[n] + heuristic(n, problem, dir)
            pr = max(f, 2 * g[n])
            prmin = min(prmin, pr)
            prmin_F = min(prmin_F, f)

        return prmin, prmin_F, min(g.values())

    def min_p_g(prmin, open1, g, dir):
        """find prmin and gmin in open list"""
        m = np.inf
        node = problem.goal
        final_path = []
        for (n, path) in open1:
            pr = max(g[n] + heuristic(n, problem, dir), 2 * g[n])
            if pr == prmin:
                if g[n] < m:
                    m = g[n]
                    node = n
                    final_path = path

        return node, final_path

    def getPath(open_fwd, open_bck):
        """Get the optimal forward and backward path"""
        for (nf, path_fwd) in open_fwd:
            for (nb, path_bck) in open_bck:
                if(nf == nb):
                    return path_fwd, path_bck
        #If no nodes are found to be common
        print('No common node found #SR')


    def opposite(path):
        """Reverse the directions in the given path. This is used for the path from
        the goal node to the start node"""
        reversed_path = []
        for i in path:
            # Convert NORTH to SOUTH
            if i == 'North':
                reversed_path.append('South')
            # Convert SOUTH to NORTH
            elif i == 'South':
                reversed_path.append('North')
            # Convert EAST to WEST
            elif i == 'East':
                reversed_path.append('West')
            # Convert WEST to EAST
            else:
                reversed_path.append('East')
        #print('\n Path_bck = {0}'.format(j))
        return reversed_path

    #while the open lists are not empty
    while open_fwd and open_bck:
        prmin_F, fmin_fwd, gmin_fwd = choose_min_n(open_fwd, g_fwd, 0)
        prmin_b, fmin_bck, gmin_bck = choose_min_n(open_bck, g_bck, 1)
        C = min(prmin_F, prmin_b)

        if U <= max(C, fmin_fwd, fmin_bck, gmin_fwd + gmin_bck + 1):
            """The condition that indicates that the optimal solution has been found.
            The cost of the cheapest edge in this problem is 1"""
            """
            totalOpenNodes = len(open_fwd) + len(open_bck) + 1
            totalClosedNodes = len(closed_fwd) + len(closed_bck)
            print('\nTotal nodes expanded = {0}'.format(totalOpenNodes + totalClosedNodes))
            print(' (open nodes = {0} and closed nodes = {1})'.format(totalOpenNodes, totalClosedNodes))
            """
            print('\nPath length = {0}'.format(U))
            path_fwd, path_bck = getPath(open_fwd, open_bck)
            #print('\n path_bck = {0}'.format(path_bck))
            path_bck = reversed(path_bck)
            #print('\n Path_fwd = {0}'.format(path_fwd))
            if path_bck:
                path_fwd= path_fwd + opposite(path_bck)
            problem.isGoalState(problem.getGoalState())
            return path_fwd

        if C == prmin_F:
            # Search in the forward direction
            U, open_fwd, closed_fwd, g_fwd = search_dir(U, open_fwd, open_bck, g_fwd, g_bck, closed_fwd, 0)
        else:
            # Search in the backward direction
            U, open_bck, closed_bck, g_bck = search_dir(U, open_bck, open_fwd, g_bck, g_fwd, closed_bck, 1)

    #Incase U never reaches the optimal value
    print('\nPath length = infinity')
    return path_fwd


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
mm = bidirectionalMMsearch
