import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw 
from rdkit.Chem import rdFMCS
import pandas as pd
from collections import deque
import logging as lg
lg.basicConfig(level=lg.DEBUG)


def insert_sublist(original_list, insert_list, insert_position):
    """
    :param original_list: original list
    :param insert_list: list to be inserted
    :param insert_position: position to insert the list

    Insert a list into another list

    :return: new list

    """

    part1 = original_list[:insert_position]
    part2 = original_list[insert_position:]
    
    part1.extend(insert_list)
    part1.extend(part2)

    return part1



def smiles2graph(smiles):
    """
    :param smiles: SMILES string
    :return: adjacency matrix and feature matrix
    """
    mol = Chem.MolFromSmiles(smiles)
    n_atoms = mol.GetNumAtoms()
    adj = Chem.rdmolops.GetAdjacencyMatrix(mol)
    fea = np.zeros((n_atoms, 1))
    for atom in mol.GetAtoms():
        fea[atom.GetIdx()] = atom.GetAtomicNum()
    return adj,fea

def graph_dict(nodes,edges,beads_info):
    """
    :param nodes: list of nodes
    :param edges: list of edges

    Convert graph to dictionary
    Check external connections of each node, and store them in the form of {node: [neighbors]}
    Replace the external connection with the corresponding subgroup

    :return: graph dictionary
    """
    graph = {}
    for u, v in edges:
        if u not in graph:
            graph[u] = []
        if v not in graph:
            graph[v] = []
        graph[u].append(v)
        graph[v].append(u)
    
    for node in graph.keys():
        for sub in beads_info.keys():
            for n in graph[node]:
                if n in beads_info[sub]['subgraph_atoms']:
                    graph[node].remove(n)
                    graph[node].append(sub)
        graph[node] = list(set(graph[node]))

    return graph

def bfs(graph, start, end):
    """
    :param graph: graph dictionary
    :param start: start node
    :param end: end node

    Breadth-first search algorithm

    :return: shortest path
    """
    visited = set()
    queue = deque([(start, [start])])

    while queue:
        node, path = queue.popleft()
        if node == end:
            return path
        if node in visited:
            continue
        visited.add(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                new_path = list(path)
                new_path.append(neighbor)
                queue.append((neighbor, new_path))


def is_subgroup_of_existing(new_subgroup_atoms, beads_info):
    """
    :param new_subgroup_atoms: list of atom indices of the new subgroup
    :param beads_info: dictionary of beads information

    Check if the new subgroup is a subset of an existing subgroup

    :return: True if the new subgroup is a subset of an existing subgroup, False otherwise
    """
    for bead in beads_info.values():
        if set(new_subgroup_atoms).issubset(set(bead['subgraph_atoms'])):
            return True
    return False

def reconstruct_graph(HT, mol, beads_info,Subgroups_dict):
    """
    :param HT: list of atom indices of the head and tail
    :param mol: rdkit mol object
    :param beads_info: dictionary of beads information

    Reconstruct the graph by considering the subgroups as nodes
    """
    non_subgroup_atoms = set(range(mol.GetNumAtoms()))
    for bead_info in beads_info.values():
        non_subgroup_atoms -= set(bead_info['subgraph_atoms'])
    #  non_subgroup_atoms -= set(HT[0])
    #  non_subgroup_atoms -= set(HT[1])

    new_nodes = list(non_subgroup_atoms) + list(beads_info.keys())
    
    #Check if node is subgroup
    connections = []
    for i in range(len(new_nodes)):
        connection = []
        if new_nodes[i] in beads_info.keys():
            connection.extend(beads_info[new_nodes[i]]['external_connections'])
        else:
            atom = mol.GetAtomWithIdx(new_nodes[i])
            for neighbor in atom.GetNeighbors():
                neighbor_idx = neighbor.GetIdx()
                if neighbor_idx in new_nodes:
                    connection.extend([neighbor_idx])
        for conn in connection:
            connections.append([new_nodes[i],conn])
    graph = graph_dict(new_nodes,connections,beads_info)
    backbone = bfs(graph, HT[1][0], HT[0][0])

    HAPPY_graph = backbone.copy()
    #print(backbone)
    new = []
    CHANGE = True
    while CHANGE == True:
        new = update_graph(HAPPY_graph,new,graph)
        if new != HAPPY_graph:
            HAPPY_graph = new.copy()
        else:
            CHANGE = False
    return HAPPY_graph

def update_graph(HAPPY_graph,new,graph):
    """
    :param HAPPY_graph: list of nodes
    :param new: list of nodes
    :param graph: graph dictionary

    Update the graph by considering the subgroups as nodes

    :return: updated graph
    """
    #  print(HAPPY_graph)
    for node in HAPPY_graph:
        if node in ['@','#']:
            continue
        connection = graph[node]
        idx = HAPPY_graph.index(node)

        #remove connection info which is already in the HAPPY_graph
        remove = []
        for conn in HAPPY_graph:
            if conn in connection:
                remove.append(conn)
        for r in remove:
            connection.remove(r)

        if len(connection) == 1:
            chunk = ['@']
            for conn in connection:
                if conn not in HAPPY_graph:
                    #  print(conn,idx)
                    chunk.append(conn)
            HAPPY_graph = insert_sublist(HAPPY_graph,chunk,idx+1)
        if len(connection) == 2:
            chunk = ['#']
            for conn in connection:
     
                if conn not in HAPPY_graph:
                    #  print(conn,idx)
                    chunk.append(conn)
            HAPPY_graph = insert_sublist(HAPPY_graph,chunk,idx+1)
    return HAPPY_graph


def subgroup(mol,sub,beads_info):
    """
    :param mol: rdkit mol object
    :param sub: SMILES string of the subgroup
    :param bead_info: dictionary of beads information

    Compute the subgraph of the molecule and the external connections of the subgraph

    :return: updated dictionary of beads information
    :bead_information contains: index of atoms in the subgraph, index of atoms connected to the subgraph
    """
    SMARTS_sub = Chem.MolToSmarts(Chem.MolFromSmiles(sub))
    matches_SMARTS = mol.GetSubstructMatches(Chem.MolFromSmarts(SMARTS_sub))
    matches_SMILES = mol.GetSubstructMatches(Chem.MolFromSmiles(sub))
    
    #compare length of matches_SMARTS and matches_SMILES
    if len(matches_SMARTS) == 0 and len(matches_SMILES) != 0:
        matches = matches_SMILES
    else:
        matches = matches_SMARTS
   
    for i, m in enumerate(matches):
        num = len(beads_info)
        if not is_subgroup_of_existing(m, beads_info):
            external_connections = []
            for atom_idx in m:
                atom = mol.GetAtomWithIdx(atom_idx)
                for neighbor in atom.GetNeighbors():
                    neighbor_idx = neighbor.GetIdx()
                    if neighbor_idx not in m:
                        external_connections.append(neighbor_idx)

            beads_info[f'S{num+1}'] = {
                'subgraph_atoms': m,
                'external_connections': external_connections,
                'name': Subgroups_dict[sub]
            }
    return beads_info

def string_encoding(HAPPY_graph,mol,beads_info):
    """
    :param HAPPY_graph: list of nodes
    :param mol: rdkit mol object
    :param beads_info: dictionary of beads information

    Encode the graph as a string

    :return: string encoding of the graph
    """
    HAPPY_graph = HAPPY_graph[1:-1]
    string = 'H'
    for node in HAPPY_graph:
        if node in ['@','#']:
            string += node
        elif type(node) == str:
            string += '*'+beads_info[node]['name']
        else:
            atom = mol.GetAtomWithIdx(node)
            string += atom.GetSymbol()
    string += 'T'
    return string

data = pd.read_csv('SMILES.csv',index_col=0)
df = pd.read_csv('subgroup.csv',index_col=0)
Subgroups_dict = dict(zip(df.smiles,df.subgroup))

#Sort Subgroup by size
Subgroups = list(Subgroups_dict.keys())
len_sub = []
for s in Subgroups:
    atoms = Chem.MolFromSmarts(s).GetAtoms()
    len_sub.append(len(atoms))

subgroup_tuples = zip(Subgroups, len_sub)

sorted_subgroup_tuples = sorted(subgroup_tuples, key=lambda x: x[1],reverse=True)
Subgroups_dict = {k: Subgroups_dict[k] for k, _ in sorted_subgroup_tuples}

Subgroups = list(Subgroups_dict.keys())

i = 2
for s in data.S0:
    beads_info = {}
    mol = Chem.MolFromSmiles(s)
    adj,fea = smiles2graph(s)
    HT = mol.GetSubstructMatches(Chem.MolFromSmiles('[*]'))
    for sub in Subgroups:
        beads_info = subgroup(mol,sub,beads_info)
    E = []
    for b in beads_info:
        E.extend(beads_info[b]['external_connections'])

    try : rec = reconstruct_graph(HT, mol, beads_info,Subgroups_dict)
    except : 
        print("==================ERROR=======================")
        print(f'{i:<{2}} -> {s:<{100}}')
        break
    HAPPY_string = string_encoding(rec,mol,beads_info)
    #print(beads_info)
    #  print(beads_info)
    #find the index where s, and query S2 data
    #  print(HAND_HAPPY)
    print(f'{i:<{2}} {s:<{100}} -> {HAPPY_string:<{30}}')
    i += 1

    #  print(s,HAPPY_string)w
