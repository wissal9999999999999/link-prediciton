import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+"/src")

import networkx as nx
import numpy as np

from models import AdamicAdar
from utils import read_graph_from_json

if __name__ == '__main__':
    num_recommendations = 20


    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct the path to go.json relative to the script directory
    data_path = os.path.join(script_dir, "../..", "out.json")

    # Read the graph from the JSON file
    graph, node_ids, edge_ids = read_graph_from_json(data_path)
    adj = nx.Graph(graph)

    name_to_node = {node_ids[i]: i for i in range(adj.number_of_nodes())}

    model = AdamicAdar()
    model.train(adj, [], [], [], [], None, (0, 0))
    preds = model.predict([(name_to_node['http://www.semanticweb.org/stagiaire1/ontologies/2016/5/untitled-ontology-49#AutoimmuneLiverDisease'], i) for i in range(adj.number_of_nodes())])
    # Get the adjacency matrix of the graph
    adj_matrix = nx.adjacency_matrix(adj)

    # Convert the adjacency matrix to an array
    adj_array = adj_matrix.toarray()

    # Calculate redundant and missing
    redundant = preds * adj_array[name_to_node['http://www.semanticweb.org/stagiaire1/ontologies/2016/5/untitled-ontology-49#AutoimmuneLiverDisease'], :]
    missing = preds - redundant

    sorted_redundant = np.argsort(redundant)
    sorted_missing = np.argsort(-missing)

    i = 0
    print("Redundant")
    for ind in sorted_redundant:
        if adj.has_edge('http://www.semanticweb.org/stagiaire1/ontologies/2016/5/untitled-ontology-49#AutoimmuneLiverDisease', node_ids[ind]) and adj_array[name_to_node['http://www.semanticweb.org/stagiaire1/ontologies/2016/5/untitled-ontology-49#AutoimmuneLiverDisease'], ind] > 0.5:
            i += 1
            print("{} - {}, Score: {}".format('http://www.semanticweb.org/stagiaire1/ontologies/2016/5/untitled-ontology-49#AutoimmuneLiverDisease', node_ids[ind], redundant[ind]))
            if i > num_recommendations:
                break
    print()

    print("Missing")
    for i in range(num_recommendations):
        print("{} - {}, Score: {}".format('http://www.semanticweb.org/stagiaire1/ontologies/2016/5/untitled-ontology-49#AutoimmuneLiverDisease', node_ids[sorted_missing[i]], missing[sorted_missing[i]]))
