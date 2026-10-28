from dtqw import DTQW, Any, Graph, Node, complexfloating, dtype, ndarray, plt


class Link_Prediction:

    def __init__(self, graph: Graph | list[list[int]] | list[Node] | ndarray[Any, dtype[complexfloating]]):
        self.dtqw = DTQW(graph)
        self.scores: list[float] = []

    def predict(self, steps: int, register_all_scores: bool, starting_node: int) -> None:

        if steps < 1:
            raise ValueError("steps must be a positive integer")

        if steps < 2:
            raise ValueError("having less than 2 steps produces useless data as all the resulting nodes are already connected to the initial one")

        if starting_node > self.dtqw._num_nodes-1 or starting_node < 0:
            raise ValueError(f"starting_node must be between 0 and {self.dtqw._num_nodes-1}")

        self.dtqw.reset()

        if register_all_scores:
            register_probabilities = "all"
        else:
            register_probabilities = "last"

        self.dtqw.simulate(steps, register_probabilities, starting_node)

        neighbors: list[int] = self.dtqw._adjacency_list[starting_node].neighbors
        starting_node_degree: complex = abs(self.dtqw.get_degree(starting_node)['out'])

        scores: list[float] = [0.0 for _ in self.dtqw._probabilities[0]]

        if register_all_scores == True:
            useful_data: list[list[float]] = self.dtqw._probabilities[1:]
        else:
            useful_data: list[list[float]] = [self.dtqw._probabilities[-1]]

        for prob_list in useful_data:

            for node in range(len(prob_list)):
                if node in neighbors or node == starting_node:
                    continue

                # probability of node j * (degree of i + degree of j)
                scores[node] += prob_list[node] * (starting_node_degree + abs(self.dtqw.get_degree(node)["out"]))

        for i in range(len(scores)):
            scores[i] /= steps-1 # ignoring first step

        self.scores = scores

    def plot_scores(self) -> None:
        plt.bar(range(len(self.scores)), self.scores)
        plt.show()



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