from discrete_walk import *

class LinkPrediction():

    def __init__(self, graph : list[list[int]] | nx.Graph):
        self.dtw_simulation = DTQW(graph)
        self.scores : list[float] = []
    
    def predict(self, steps : int, register_all_scores : bool, starting_node : int) -> None:

        if steps < 1:
            raise ValueError("steps must be a positive integer")

        if steps < 2:
            raise ValueError("having less than 2 steps produces useless data as all the resulting nodes are already connected to the initial one")

        if starting_node > self.dtw_simulation.num_nodes-1:
            raise ValueError(f"starting_node must be between 0 and {self.dtw_simulation.num_nodes-1}")

        self.dtw_simulation.reset()

        if register_all_scores:
            register_probabilities = "all"
        else:
            register_probabilities = "last"

        self.dtw_simulation.simulate(
            steps=steps,
            register_probabilities=register_probabilities,
            starting_node=starting_node
        )

        adjacent_nodes = self.dtw_simulation._nodes[starting_node].connected_nodes
        starting_node_degree = self.dtw_simulation._nodes[starting_node].degree

        scores = [0.0 for _ in self.dtw_simulation.probabilities[0]]

        if register_all_scores == True:
            useful_data = self.dtw_simulation.probabilities[1:]
        else:
            useful_data = self.dtw_simulation.probabilities

        for prob_list in useful_data:

            for node in range(len(prob_list)):
                if node in adjacent_nodes or node == starting_node:
                    continue

                # probability of node j * (degree of i + degree of j)
                scores[node] += prob_list[node] * (starting_node_degree + self.dtw_simulation._nodes[node].degree)

        for i in range(len(scores)):
            scores[i] /= steps-1 # ignoring first step

        self.scores = scores

    def plotScores(self):
        plt.bar(range(len(self.scores)), self.scores)
        plt.show()

if __name__ == "__main__":
    simple_4x4 = [
        [0, 1, 1, 1],
        [1, 0, 1, 0],
        [1, 1, 0, 0],
        [1, 0, 0, 0]
    ]

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

    lp = LinkPrediction(study_matrix)
    #lp.predict(steps=2, register_all_scores=True, starting_node=5)
    #lp.plotScores()
    lp.predict(steps=3, register_all_scores=True, starting_node=5)
    lp.plotScores()