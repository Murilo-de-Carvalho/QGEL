from qgel.dtqw import DTQW, Any, Graph, Node, complexfloating, dtype, ndarray, plt


class Link_Prediction:
    """
    Class for Link Prediction using DTQW.

    Using a DTQW, starting from a specified node and a step count, all other nodes receive a "link score".
    High scores indicate that the two nodes should be linked.

    Parameters
    ----------
    graph : Graph | list[list[int]] | list[Node] | ndarray[Any, dtype[complexfloating]]
        Representation of the graph you want to work on.

    Examples
    --------
    Creating a link prediction class:

    >>> adj_matrix = [
    >>>     [0, 1, 1],
    >>>     [1, 0, 0],
    >>>     [0, 1, 0]
    >>> ]
    >>> lp = Link_Prediction(adj_matrix)

    From a nx Graph:

    >>> G = nx.gnp_random_graph(100, 0.3) # Random graph with 100 nodes
    >>> lp = DTQW(G)

    **Methods:**

    Predicting link, starting from node 2, registering all scores (not only the final one):

    >>> lp.predict(steps=10, register_all_scores=True, starting_node_id=2)

    Plotting the scores:

    >>> lp.plot_scores()

    Raises
    ------
    ValueError
        When inputting a type not supported as a graph representation.
    """
    
    def __init__(self, graph: Graph | list[list[int]] | list[Node] | ndarray[Any, dtype[complexfloating]]):
        self.dtqw = DTQW(graph)
        self.scores: list[float] = []

    def predict(self, steps: int, register_all_scores: bool, starting_node_id: int) -> None:
        """
        Runs a DTQW, starting from the starting node selected.
        
        If register_all_scores is True, at each step calculate the score of
        every node and, at the, takes the median of thse scores.

        If register_all_scores is False, at the end of the walk, calculate the scores of
        every node and that is the final value.

        Parameters
        ----------
        steps : int
            Amount of steps given to the DTQW, registering the scores of every node for each step if
            register_all_scores is True.

        register_all_scores : bool
            If the scores should be only the final result or a median of all results, one for each step.

        starting_node_id : int
            Defines the starting point of the DTQW.

        Raises
        ------
        ValueError
            When steps is a non-positive value

        ValueError
            When steps is less than 2. It would produce useless data

        ValueError
            When starting_node_id is out of the range 0 ... self.dtqw._num_nodes-1
        """

        if steps < 1:
            raise ValueError("steps must be a positive integer")

        if steps < 2:
            raise ValueError("having less than 2 steps produces useless data as all the resulting nodes are already connected to the initial one")

        if starting_node_id > self.dtqw._num_nodes-1 or starting_node_id < 0:
            raise ValueError(f"starting_node_id must be between 0 and {self.dtqw._num_nodes-1}")

        self.dtqw.reset()

        if register_all_scores:
            register_probabilities = "all"
        else:
            register_probabilities = "last"

        self.dtqw.simulate(steps, register_probabilities, starting_node_id)

        neighbors: list[int] = self.dtqw._adjacency_list[starting_node_id].neighbors
        starting_node_id_degree: complex = abs(self.dtqw.get_degree(starting_node_id)['out'])

        scores: list[float] = [0.0 for _ in self.dtqw._probabilities[0]]

        if register_all_scores == True:
            useful_data: list[list[float]] = self.dtqw._probabilities[1:]
        else:
            useful_data: list[list[float]] = [self.dtqw._probabilities[-1]]

        for prob_list in useful_data:

            for node in range(len(prob_list)):
                if node in neighbors or node == starting_node_id:
                    continue

                # probability of node j * (degree of i + degree of j)
                scores[node] += prob_list[node] * (starting_node_id_degree + abs(self.dtqw.get_degree(node)["out"]))

        for i in range(len(scores)):
            scores[i] /= steps-1 # ignoring first step

        self.scores = scores

    # TODO: Make this better
    def plot_scores(self) -> None:
        """
        Plots the scores of each node, except the starting node, using plt.
        """
        
        plt.bar(range(len(self.scores)), self.scores)
        plt.show()

    # TODO: Create reset()



if __name__ == "__main__":

    study_matrix = [
        [0, 0, 1, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [1, 0, 0, 0, 1, 0, 1, 0],
        [0, 0, 0, 0, 1, 1, 0, 0],
        [0, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 1, 0]
    ]

    lp = Link_Prediction(study_matrix)
    lp.predict(20, True, 4)
    lp.plot_scores()
    lp.dtqw.draw()