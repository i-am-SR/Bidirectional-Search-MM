# Bidirectional-Search-MM
Implementation of bidirectional search that meets in the middle in the Pacman domain

Implemented bi-directional search described in the following paper: “Bidirectional Search That Is Guaranteed to Meet   
in the Middle”, Robert C. Holte, Ariel Felner, Guni Sharon, Nathan R. Sturtevant, AAAI 2016, and integrated it into the   
Pacman domain for path-finding problems (from start to a fixedgoal location)  
(http://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/download/12320/12109)  


Sample run on bogMaze with Manhattan heuristic:

![Alt Text](https://github.com/i-am-SR/Bidirectional-Search-MM/blob/master/gif/Screencast%20from%2005-13-2020%2001_40_15%20PM.gif)


To run the MM search:

manhattan heuristic=> python pacman.py -l mediumMaze -z .5 -p SearchAgent -a fn=mm,heuristic=manhattanHeuristic_bi

euclidean heuristic=> python pacman.py -l mediumMaze -z .5 -p SearchAgent -a fn=mm,heuristic=euclideanHeuristic_bi

null heuristic=> python pacman.py -l mediumMaze -z .5 -p SearchAgent -a fn=mm,heuristic=nullHeuristic_bi -q


To run the A* search:

manhattan heuristic=> python pacman.py -l mediumMaze -z .5 -p SearchAgent -a fn=astar,heuristic=manhattanHeuristic -q

euclidean heuristic=> python pacman.py -l mediumMaze -z .5 -p SearchAgent -a fn=astar,heuristic=euclideanHeuristic -q

null heuristic=> python pacman.py -l mediumMaze -z .5 -p SearchAgent -a fn=astar,heuristic=nullHeuristic -q


To run the BFS search:

python pacman.py -l mediumMaze -z .5 -p SearchAgent -a fn=bfs -q


To run the DFS search:

python pacman.py -l mediumMaze -z .5 -p SearchAgent -a fn=dfs -q


The name of the layout used has to be mentioned after the -l attribute. The layouts are present in the layouts folder.

