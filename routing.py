import csv
import sys
import os

# Reads the network topology from a given CSV file
def read_topology(filename):
    # Open the specified CSV file 
    with open(filename, 'r') as infile:
        # Create a CSV reader object to parse the file
        reader = csv.reader(infile)
        # Skip the first row of headers. This is under the assumption the first column is not part of the data
        headers = next(reader)[1:]  
        # Initialize a dictionary to hold the topology data
        # Each header (node) is a key, mapping to another dictionary representing connections
        topology = {header: {} for header in headers}
        # Iterate over each row in the CSV file
        for row in reader:
            # The first element of each row is treated as the row header (node identifier)
            row_header = row[0]
            # Iterate over each cell in the row alongside its corresponding header
            # Skips the first cell as it's the row header
            for header, cell in zip(headers, row[1:]):
                # Check if the cell value is not '9999' (indicating no connection)
                if cell != "9999":  
                    # Add a connection with its cost to the topology
                    # Convert cell value to float assuming it represents a numerical cost
                    topology[row_header][header] = float(cell)
    # Return the populated topology dictionary
    return topology


# Implements Dijkstra's algorithm 
def dijkstra_algorithm(topology, source_node):
    # Initialize all nodes as unvisited with an infinite cost, except the source node
    # This is to set the initial state where we don't know the shortest paths yet
    unvisited = {node: float('inf') for node in topology}
    # Cost to reach the source node is always 0
    unvisited[source_node] = 0  
    # A dictionary to store the visited nodes and their cost
    visited = {}
    # Dictionary to keep track of the path taken to reach each node
    path = {}
    # Loop until all nodes are visited
    while unvisited:
        # Select the unvisited node with the smallest cost
        min_node = min(unvisited, key=unvisited.get)
        visited[min_node] = unvisited[min_node]
        # Update the cost of the neighbouring nodes of the current node
        for neighbour, cost in topology[min_node].items():
            # Ensure the neighbour has not been visited yet
            if neighbour not in visited:
                # Calculate new cost to reach this neighbour
                new_cost = unvisited[min_node] + cost
                # If the new cost is less than the current cost in unvisited, update it
                # This step ensures that we find the shortest path to the neighbour
                if new_cost < unvisited[neighbour]:
                    unvisited[neighbour] = new_cost
                    # Update the path to reflect that the best way to reach this neighbour is via the current node
                    path[neighbour] = min_node
        # Mark the current node as visited by removing it from unvisited
        unvisited.pop(min_node)
    # Return the path dictionary, showing the optimal path to each node, and the visited dictionary, containing the cost to reach each node
    return path, visited


# Implements the distance vector algorithm using Bellman-Ford equation.
def distance_vector_algorithm(topology):
    # Initialize the distance vectors with infinite costs for all nodes
    # This creates a dictionary of dictionaries, where each node has a distance vector to every other node
    distance_vectors = {node: {n: float('inf') for n in topology} for node in topology}
    # Set the distance to itself as 0 for every node
    for node in topology:
        distance_vectors[node][node] = 0
    # Loop to update the distance vector for each node
    # The loop runs 'number of nodes - 1' times, as per Bellman-Ford logic
    for _ in range(len(topology) - 1):
        # Iterate over each node in the topology
        for node in topology:
            # Iterate over each neighbour of the current node and the cost to reach that neighbour
            for neighbour, cost in topology[node].items():
                # Update the distance vector using the Bellman-Ford equation
                for n in topology:
                    # The distance to a node 'n' is the minimum of the current known distance and the distance to a neighbour plus the cost from that neighbour to 'n'
                    distance_vectors[node][n] = min(distance_vectors[node][n], cost + distance_vectors[neighbour][n])
    # Return the final distance vectors for all nodes
    # Each node's distance vector represents the shortest known paths to every other node
    return distance_vectors


# Helper functions to format the output correctly
def format_costs(cost_dict, source_node):
    # Initialize the formatted string with the cost to the source node itself, which is always 0
    formatted_costs = f'{source_node}:0'
    # Iterate over each node and its cost in the cost dictionary, sorted for consistency
    for node, cost in sorted(cost_dict.items()):
        if node != source_node:
            # Check if the cost is an integer to format it correctly (as int or float)
            cost_str = str(int(cost)) if cost.is_integer() else str(cost)
            # Append this node's cost to the formatted string
            formatted_costs += f', {node}:{cost_str}'
    # Return the fully formatted cost string
    return formatted_costs


# Helper functions to format the output correctly
def format_distance_vector(distance_vector):
    # Initialize an empty list to hold formatted distance values
    formatted_vector_elements = []
    # Iterate over each node in the sorted distance vector
    for node in sorted(distance_vector.keys()):
        # Get the distance value for this node
        distance = distance_vector[node]
        # Check if the distance value is an integer to format it correctly
        distance_str = str(int(distance)) if distance.is_integer() else str(distance)
        # Append the formatted distance to the list
        formatted_vector_elements.append(distance_str)
    # Join all formatted elements into a single string separated by spaces
    formatted_vector = ' '.join(formatted_vector_elements)
    # Return the formatted distance vector string
    return formatted_vector


# Helper functions to format the output correctly
def format_path(path_dict, source_node):
    # Initialize an empty list to hold all formatted paths
    paths = []
    # Iterate over each node in the path dictionary, sorted to maintain consistent order
    for target_node in sorted(path_dict.keys()):
        # Ensure the target node is not the source node, as we are looking for paths to other nodes
        if target_node != source_node:
            # Initialize the path list with the target node
            path = [target_node]
            # Trace back the path from the target node to the source node
            while target_node in path_dict:
                # Get the next node in the path towards the source node
                target_node = path_dict[target_node]
                # Add this node to the path list
                path.append(target_node)
            # Check if the last node in the path is the source node
            # This confirms a complete path from source to target
            if path[-1] == source_node:
                # Add the reversed path (to display from source to target) to the list of paths
                paths.append(''.join(reversed(path)))
    # Sort the paths by their length to ensure shorter paths are listed first
    # If lengths are equal, sort alphabetically
    paths.sort(key=lambda x: (len(x), x))
    # Return a single string with all paths separated by space
    return ' '.join(paths)


# The main function to execute the program.
def main():

    # Check for correct command line arguments with error message on proper usage
    if len(sys.argv) != 2:
        print("Usage: python routing.py <topology.csv>")
        sys.exit()
    topology_file = sys.argv[1]
    # Check if the topology file exists
    if not os.path.exists(topology_file):
        print("File {} not found.".format(topology_file))
        sys.exit()

    # Read the topology from the file
    topology = read_topology(topology_file)
    # Get the source node from user input
    source_node = input("Please provide the source node: ")
    # Run Dijkstra's algorithm to find the shortest path and costs
    path, costs = dijkstra_algorithm(topology, source_node)
    # Print the shortest path tree for the source node
    print("Shortest path tree for node {}:\n{}".format(source_node, format_path(path, source_node)))
    # Print the costs of the least-cost paths for the source node
    print("Costs of the least-cost paths for node {}:\n{}".format(source_node, format_costs(costs, source_node)))
    
    # Print the distance vectors
    print()
    distance_vectors = distance_vector_algorithm(topology)
    for node in sorted(distance_vectors.keys()):
        formatted_vector = format_distance_vector(distance_vectors[node])
        print("Distance vector for node {}: {}".format(node, formatted_vector))


if __name__ == "__main__":
    main()
