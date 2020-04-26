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
    curr_state_fwd = problem.getStartState()
    curr_state_bck = problem.getGoalState()
    visited_states = set()  # to store the states already visited
    visited_states.add(curr_state_fwd)
    visited_states.add(curr_state_bck)
    pathF = []
    pathB = []
    gF, gB = {curr_state_fwd: 0}, {curr_state_bck: 0}
    openF, openB = [(curr_state_fwd, pathF)], [(curr_state_bck, pathB)]
    closedF, closedB = [], []
    U = np.inf

    def extend(U, open_dir, open_other, g_dir, g_other, closed_dir, dir):
        n, path = find_key(C, open_dir, g_dir, dir)

        open_dir.remove((n, path))
        closed_dir.append((n, path))

        successor_list = problem.getSuccessors(n)
        for (c, next_direction, additional_cost) in successor_list:
            if found(open_dir, c) or found(closed_dir, c):
                if g_dir[c] <= g_dir[n] + additional_cost:
                    continue

                open_dir = delete(open_dir, c)

            g_dir[c] = g_dir[n] + additional_cost
            open_dir.append((c, path + [next_direction]))
            visited_states.add(c)

            if found(open_other, c):
                U = min(U, g_dir[c] + g_other[c])

        return U, open_dir, closed_dir, g_dir

    def delete(open_dir, n):
        """Check if the state n is on the Open list open_dir"""
        for (c, path) in open_dir:
            if c == n:
                open_dir.remove((c, path))
        return open_dir

    def found(open_dir, n):
        """Check if the state n is on the Open list open_dir"""
        for (c, path) in open_dir:
            if c == n:
                return True
        return False

    def find_min(open_dir, g, dir):
        """Finds minimum priority, g and f values in open_dir"""
        # pr_min_f isn't forward pr_min instead it's the f-value
        # of node with priority pr_min.
        pr_min, pr_min_f = np.inf, np.inf
        for (n, path) in open_dir:
            f = g[n] + heuristic(n, problem, dir)
            pr = max(f, 2 * g[n])
            pr_min = min(pr_min, pr)
            pr_min_f = min(pr_min_f, f)

        return pr_min, pr_min_f, min(g.values())

    def find_key(pr_min, open_dir, g, dir):
        """Finds key in open_dir with value equal to pr_min
        and minimum g value."""
        m = np.inf
        node = problem.goal
        final_path = []
        for (n, path) in open_dir:
            pr = max(g[n] + heuristic(n, problem, dir), 2 * g[n])
            if pr == pr_min:
                if g[n] < m:
                    m = g[n]
                    node = n
                    final_path = path

        return node, final_path

    def getPath(openF, openB):
        """Get the optimal forward and backward path"""
        for (nf, pathF) in openF:
            for (nb, pathB) in openB:
                if(nf == nb):
                    return pathF, pathB
        print('No common node found #SR')


    def opposite(path):
        """
        Reverses the path followed from the goal node towards the start node
        """
        j = []
        for i in path:
            # Convert NORTH to SOUTH
            if i == 'North':
                j.append('South')
            # Convert SOUTH to NORTH
            elif i == 'South':
                j.append('North')
            # Convert EAST to WEST
            elif i == 'East':
                j.append('West')
            # Convert WEST to EAST
            else:
                j.append('East')
        #print('\n PathB = {0}'.format(j))
        return j

    while openF and openB:
        pr_min_f, f_min_f, g_min_f = find_min(openF, gF, 0)
        pr_min_b, f_min_b, g_min_b = find_min(openB, gB, 1)
        C = min(pr_min_f, pr_min_b)

        if U <= max(C, f_min_f, f_min_b, g_min_f + g_min_b + 1):
            """
            totalOpenNodes = len(openF) + len(openB) + 1
            totalClosedNodes = len(closedF) + len(closedB)
            print('\nTotal nodes expanded = {0}'.format(totalOpenNodes + totalClosedNodes))
            print(' (open nodes = {0} and closed nodes = {1})'.format(totalOpenNodes, totalClosedNodes))
            """
            print('\nPath length = {0}'.format(U))
            pathF, pathB = getPath(openF, openB)
            #print('\n PathB = {0}'.format(pathB))
            pathB = reversed(pathB)
            #print('\n PathF = {0}'.format(pathF))
            if pathB:
                pathF= pathF + opposite(pathB)
            return pathF

        if C == pr_min_f:
            # Extend forward
            U, openF, closedF, gF = extend(U, openF, openB, gF, gB, closedF, 0)
        else:
            # Extend backward
            U, openB, closedB, gB = extend(U, openB, openF, gB, gF, closedB, 1)

    print('\nPath length = infinity')
    return pathF



'''
def bidirectionalMMSearch(problem, heuristic=nullHeuristic):
	    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    curr_state_fwd = problem.getStartState()
    curr_state_bck = problem.getGoalState()
    
    path_fwd = []
    path_bck = []
    curr_cost_fwd = 0
    curr_cost_bck = 0

    open_fwd = set()	#to store the states already visited while going forward
    closed_fwd = set()	#to store the states already visited while going forward
    open_bck = set()	#to store the states already visited while going backward
    closed_bck = set()	#to store the states already visited while going backward
    open_fwd.add(curr_state_fwd)
    open_bck.add(curr_state_bck)

    pQ_open_fwd = util.PriorityQueue()	#fringe for A* is a Priority queue
    pQ_open_bck = util.PriorityQueue()	#fringe for A* is a Priority queue
    pQ_open_fwd.push((curr_state_fwd, path_fwd, curr_cost_fwd), max(curr_cost_fwd + heuristic(curr_state_fwd, problem), 2 * curr_cost_fwd))
    pQ_open_bck.push((curr_state_bck, path_bck, curr_cost_bck), max(curr_cost_bck + heuristic(curr_state_bck, problem), 2 * curr_cost_bck))
    
    pQ_f_fwd = util.PriorityQueue()	#fringe for A* is a Priority queue
    pQ_f_bck = util.PriorityQueue()	#fringe for A* is a Priority queue
    pQ_f_fwd.push((curr_state_fwd, path_fwd, curr_cost_fwd), curr_cost_fwd + heuristic(curr_state_fwd, problem))
    pQ_f_bck.push((curr_state_bck, path_bck, curr_cost_bck), curr_cost_bck + heuristic(curr_state_bck, problem))

	pQ_g_fwd = util.PriorityQueue()	#fringe for A* is a Priority queue
    pQ_g_bck = util.PriorityQueue()	#fringe for A* is a Priority queue
    pQ_g_fwd.push((curr_state_fwd, path_fwd, curr_cost_fwd), curr_cost_fwd)
    pQ_g_bck.push((curr_state_bck, path_bck, curr_cost_bck), curr_cost_bck)
    
    U = float('inf')

    
    while not pQ_open_fwd.isEmpty() and not pQ_b.isEmpty():	
    	(prmin_fwd, _, _) = pQ_open_fwd.peek()
    	(prmin_bck, _, _) = pQ_open_bck.peek()
    	(fmin_fwd, _, _) = pQ_f_fwd.peek()
    	(fmin_bck, _, _) = pQ_f_bck.peek()
    	(gmin_fwd, _, _) = pQ_g_fwd.peek()
    	(gmin_bck, _, _) = pQ_g_bck.peek()

    	C = min(prmin_fwd, prmin_bck)

    	
    	if U <= max(C, fmin_fwd, fmin_bck, gmin_fwd + gmin_bck + 1):
    		totalOpenNodes = len(open_fwd) + len(open_bck) + 1
    		totalClosedNodes = len(closed_fwd) + len(closed_bck)
    		print('\nTotal nodes expanded = {0}'.format(totalOpenNodes + totalClosedNodes))
    		print(' (open nodes = {0} and closed nodes = {1})'.format(totalOpenNodes, totalClosedNodes))
    		print('\nPath length = {0}'.format(U))
    		return path_fwd
    		#change this

    	if C == prmin_fwd:
    		#Expand in the forward direction
    		n = pQ_open_fwd.pop()
    		
    		open_fwd.remove(n[0])
    		closed_fwd.add(n[0])
    		
    		pQ_f_fwd.delete(n)
    		pQ_g_fwd.delete(n)

    		
    		successor_list_fwd = problem.getSuccessors(curr_state_fwd)
    		for (successor_state, next_direction, additional_cost) in successor_list_fwd:
    			
    			if successor_state in open_fwd.union(closed_fwd):
    				if 





    	if curr_state not in visited_states:
    		visited_states.add(curr_state)	#mark state as visited
    		successor_list = problem.getSuccessors(curr_state)	#get successors
    		for (successor_state, next_direction, additional_cost) in successor_list:	
    			pQueue.push((successor_state, path + [next_direction], curr_cost + additional_cost), curr_cost + additional_cost + heuristic(successor_state, problem))
    	(curr_state, path, curr_cost) = pQueue.pop()
    return path
    #util.raiseNotDefined()
'''

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
mm = bidirectionalMMsearch
