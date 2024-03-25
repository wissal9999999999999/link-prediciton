import os
import sys
import json
import networkx as nx
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+"/src")

from models import PreferentialAttachment
from utils import read_graph_from_json

if __name__ == '__main__':
    num_recommendations = 20

    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct the path to go.json relative to the script directory
    data_path = os.path.join(script_dir, "../..", "out-with-missing-links.json")

    # Read the graph from the JSON file
    graph, node_ids, edge_ids = read_graph_from_json(data_path)
    adj = nx.Graph(graph)

    # Convert the graph to an adjacency matrix
    adj_array = nx.to_numpy_array(adj)
    
    name_to_node = {node_ids[i]: i for i in range(adj.number_of_nodes())}

    model = PreferentialAttachment()
    model.train(adj, [], [], [], [], None, (0, 0))

    # Load the existing data
    with open(data_path, 'r') as f:
        data = json.load(f)

    # Iterate over all nodes in the graph
    for node in node_ids:
        preds = model.predict([(node, i) for i in range(adj.number_of_nodes())])

        # Calculate redundant and missing
        redundant = preds * adj_array[node, :]
        missing = preds - redundant

        sorted_redundant = np.argsort(redundant)
        sorted_missing = np.argsort(-missing)

        i = 0
        print("Redundant")
        for ind in sorted_redundant:
            if adj.has_edge(node, node_ids[ind]) and adj_array[name_to_node[node], ind] > 0.5:
                i += 1
                print("{} - {}, Score: {}".format(node_ids[node], node_ids[ind], redundant[ind]))
                if i > num_recommendations:
                    break
        print()

        print("Missing")
        missing_links = []
        for i in range(num_recommendations):
            score = missing[sorted_missing[i]]
            print("{} - {}, Score: {}".format(node_ids[node], node_ids[sorted_missing[i]], missing[sorted_missing[i]]))
            if score >= 0.5:
                missing_link = {
                    "sub": node_ids[node],
                    "pred": "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
                    "obj": node_ids[sorted_missing[i]],
                    "score": score
                }
                missing_links.append(missing_link)

        # Add the missing links to the edges
        for link in missing_links:
            data['graphs']['edges'].append({
                "sub": link['sub'],
                "pred": link['pred'],
                "obj": link['obj']
            })

    # Write the updated data to a new file
    with open('out-with-missing-links.json', 'w') as f:
        json.dump(data, f, indent=3)