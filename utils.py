import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import networkx as nx
import pyvolve
from joblib import Parallel, delayed

def get_delta_matrix(filename, sheetname):
    '''
    Function to extract the delta matrix from a sheet of the excel file 'aadelta_matrices.xlsx'.
    Note that this function is very specific to the way data is structured in this file. It will not work correclty for other files. 
    ''' 
    def _extract_aa_code(col_name):
        return col_name.split('(')[1].split(')')[0]
    AA_property = pd.read_excel(filename, sheet_name=sheetname).iloc[21:,:]
    assert AA_property.isnull().sum().sum() == 1, "Number of NA values not equal to 1"
    AA_property.fillna('AA',inplace=True)
    AA_property.columns = range(21)
    AA_property.set_index(0, inplace=True)
    AA_property =  AA_property.transpose().set_index('AA').astype(float)
    AA_property.columns = [_extract_aa_code(item) for item in AA_property.columns]
    AA_property.index = [_extract_aa_code(item) for item in AA_property.index]
    return AA_property.reindex(sorted(AA_property.columns), axis=1).reindex(sorted(AA_property.index))


class delta_matrices:
    '''
    A class that store s the delta values among 20 amino acids for all the 4 properties
    Attributes = volume, polarity, composition, aromaticity. 
    These must all be pandas dataframes of size 20 X 20
    '''
    def __init__(self, volume, polarity, composition, aromaticity):
        self.volume = volume
        self.polarity = polarity
        self.aromaticity = aromaticity
        self.composition = composition
    
    def __repr__(self):
        return f"Delta matrices of sizes  \n"\
               f"Volume      - {self.volume.shape}\n"\
               f"Polarity    - {self.polarity.shape}\n"\
               f"Aromaticity - {self.aromaticity.shape}\n"\
               f"Composition - {self.composition.shape}\n\n"\
               f"Rows and columns for Volume\n"\
               f"Rows    = {list(self.volume.index)}\n"\
               f"Columns = {list(self.volume.columns)}"
            
    def plot(self):
        fig, axs = plt.subplots(2, 2, figsize = (16,10))
        axs = axs.ravel()
        for i, a_property in enumerate(['volume', 'polarity', 'composition', 'aromaticity']):
            delta_df = getattr(self,a_property)
            sns.heatmap(np.tril(delta_df),cmap = 'GnBu',vmax = 1., square=True, cbar_kws={"shrink": .5}, ax = axs[i] );
            axs[i].set_xticklabels(labels= delta_df.columns)
            axs[i].set_yticklabels(labels= delta_df.index, rotation = 'horizontal')
            axs[i].set_title(a_property)
            
    def deltas_between_AA(self, i, j):
        assert (type(i) == str and type(j)==str), "Arguments must be strings"
        to_return = {}
        to_return['volume'] = self.volume[i][j]
        to_return['polarity'] = self.polarity[i][j]
        to_return['composition'] = self.composition[i][j]
        to_return['aromaticity'] = self.aromaticity[i][j]
        return to_return
    
    def deltas_between_seqs(self, seq1, seq2):
        '''
        Returns a pandas dataframe of all the delta values between each pair of amino acids at the same position 
        in the two argument sequences for all properties.
        '''
        assert len(seq1) == len(seq2), "Lengths of sequences must be the same"
        df_to_return = pd.DataFrame(index=['volume', 'polarity', 'composition', 'aromaticity'], columns=range(len(seq1)))
        for i, _ in enumerate(seq1):
            df_to_return[i] = pd.Series(self.deltas_between_AA(seq1[i], seq2[i]))
        return df_to_return
    
    
class clonal_graph:
    
    def __init__(self, shm_file, seqs_file):
        self.shm_file = shm_file
        self.seqs_file = seqs_file
        self.simulations = []
        
        def get_nxgraph_from_shm_file(shm_filename):
            shm_df = pd.read_csv(shm_filename, sep = '\t')
            edges = set()
            for edge_list in shm_df.Edges:
                edge_splits = edge_list.split(',')
                for e in edge_splits:
                    to_add = tuple([ 'Node'+ item + 'a'  for item in e.split('-') ])
                    edges.add(to_add)

            G = nx.DiGraph(directed= True)
            G.add_edges_from(edges)
            root_name = [n for n,d in G.in_degree() if d==0][0]
            bfs_edges = list(nx.bfs_edges(G, root_name))
            return G,root_name, bfs_edges

        def extract_root_sequence(root_name, a_seqs_file):
            seqs_df = pd.read_csv(a_seqs_file, sep = '\t', index_col=0)
            root_index = int(root_name.split('Node')[1].split('a')[0])
            return seqs_df.loc[root_index]['AA_seq']

        def get_pyvolve_phylogeny_from_nxgraph(G, root_seq):
            tree_newick = networkx_to_newick(G)
            scale_tree = 1/len(root_seq)
            tree = pyvolve.read_tree(tree =tree_newick.replace('a', 'a:'+str(scale_tree)))
            return tree 
        
        self.nx_graph, self.root_name, self.edges = get_nxgraph_from_shm_file(self.shm_file)
        self.root_sequence = extract_root_sequence(self.root_name, self.seqs_file)
        self.pyvolve_phylogeny = get_pyvolve_phylogeny_from_nxgraph(self.nx_graph, self.root_sequence)
        
        
    def sims_to_null_dists_serially(self, a_delta_matrices_instance):
        assert len(self.simulations) !=0, "No simulation data present in the clonal_graph object"
        test_stats_dfs = []
        for i, sim_seqs in enumerate(self.simulations):
            a_df = get_test_stats(sim_seqs, self.edges, a_delta_matrices_instance).add_suffix('_sim'+str(i))
            test_stats_dfs.append(a_df)

        return pd.concat(test_stats_dfs, axis = 1)
    
    def sims_to_null_dists_parallel(self, a_delta_matrices_instance, cpus, return_mean_stats = False ):
        assert len(self.simulations) !=0, "No simulation data present in the clonal_graph object"
        test_stats_dfs= Parallel(n_jobs=cpus)(delayed(get_test_stats)(sim_seqs, self.edges, a_delta_matrices_instance, return_mean_stats) \
                           for sim_seqs in self.simulations)
        return pd.concat(test_stats_dfs, axis = 1)

    def get_observed_stats(self, a_delta_matrices_instance, return_mean_stats = False):
        shm_df =  pd.read_csv(self.shm_file, sep = '\t')
        add_deltas = shm_df.apply(lambda x : x['Multiplicity'] *\
                                    pd.Series(a_delta_matrices_instance.deltas_between_AA(x['Src_AA'], x['Dst_AA'])), axis = 1)
        shm_df_with_deltas = pd.concat([shm_df, add_deltas], axis=1)
        observed_stats = shm_df_with_deltas.groupby(['Region', 'Position'])[['Multiplicity',
                                                                             'volume', 'polarity', 
                                                                             'composition', 'aromaticity']].sum()
        if not return_mean_stats:
            return observed_stats
        else:
            return observed_stats.apply(lambda x : x/x['Multiplicity'], axis = 1)


    
def get_test_stats(simulated_seqs, edge_list, a_delta_matrices_instance, return_mean_stats ):
    
    delta_dfs = []
    for src, dst in edge_list:
        delta_dfs.append(a_delta_matrices_instance.deltas_between_seqs(simulated_seqs[src], simulated_seqs[dst]))
    sum_stats = sum(delta_dfs)
    if not return_mean_stats: return sum_stats
    
    mutations = get_num_mutations(simulated_seqs, edge_list)
    multiple_mutations = mutations[mutations>1]
    if len(multiple_mutations) == 0: return sum_stats
    
    non_zero_mean_stats = sum_stats
    for index, value in multiple_mutations.items():
        non_zero_mean_stats[index] =  (non_zero_mean_stats[index])/value
    
    return non_zero_mean_stats
        
    
def get_num_mutations(seqs_dict, edge_list):
    
    def diff_between_seqs(seq1, seq2):
        assert len(seq1) == len(seq2), "Lengths of sequences must be the same"
        diff_series = pd.Series(index=range(len(seq1)))
        for i, _ in enumerate(seq1):
            diff_series[i] =  (seq1[i] != seq2[i])
        return diff_series.astype(int)

    mutations_bw_two_seqs = []
    for src, dst in edge_list:
        mutations_bw_two_seqs.append(diff_between_seqs(seqs_dict[src], seqs_dict[dst]))
    return sum(mutations_bw_two_seqs)


def networkx_to_newick(nx_graph):
    def bfs_edge_lst(graph, n):
        return list(nx.bfs_edges(graph, n))
    
    def recursive_search(a_dict, key):
        if key in a_dict:
            return a_dict[key]
        for k, v in a_dict.items():
            item = recursive_search(v, key)
            if item is not None:
                return item

    def tree_from_edge_lst(elst, root_node):
        tree = {root_node: {}}
        for src, dst in elst:
            subt = recursive_search(tree, src)
            subt[dst] = {}
        return tree

    def tree_to_newick(tree):
        items = []
        for k in tree.keys():
            s = ''
            if len(tree[k].keys()) > 0:
                subt = tree_to_newick(tree[k])
                if subt != '':
                    s += '(' + subt + ')'
            s += k
            items.append(s)
        return ','.join(items)

    root_node = [n for n,d in nx_graph.in_degree() if d==0][0]
    elst =  bfs_edge_lst(nx_graph, root_node)
    tree = tree_from_edge_lst(elst, root_node)
    newick_string = tree_to_newick(tree) + ';'
    return newick_string


def plot_nx_graph(G):
    plt.figure(figsize = (8,6))
    pos = nx.kamada_kawai_layout(G)
    nx.draw_networkx_nodes(G, pos,  node_size = 50)
    nx.draw_networkx_labels(G, pos, font_color = 'black', font_size = 8)
    nx.draw_networkx_edges(G, pos, edgelist=G.edges, edge_color='blue', arrows=True)
    plt.show()
    