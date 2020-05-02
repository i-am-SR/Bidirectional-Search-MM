# Bidirectional-Search-MM
Implementation of bidirectional search that meets in the middle in the Pacman domain

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
