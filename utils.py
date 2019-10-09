import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import networkx as nx
import pyvolve
from joblib import Parallel, delayed

########################################################################################################################
########################################## Delta Matrices class ########################################################
########################################################################################################################.

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
        '''
        Plot the delta matrices as heatmaps.
        '''
        fig, axs = plt.subplots(2, 2, figsize = (16,10))
        axs = axs.ravel()
        for i, a_property in enumerate(['volume', 'polarity', 'composition', 'aromaticity']):
            delta_df = getattr(self,a_property)
            sns.heatmap(np.tril(delta_df),cmap = 'GnBu',vmax = 1., square=True, cbar_kws={"shrink": .5}, ax = axs[i] );
            axs[i].set_xticklabels(labels= delta_df.columns)
            axs[i].set_yticklabels(labels= delta_df.index, rotation = 'horizontal')
            axs[i].set_title(a_property)
            
    def deltas_between_AA(self, i, j):
        '''
        Get delta values for 4 properties between amino acid i and j. 
        '''
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

########################################################################################################################
########################################## Clonal Graph class ##########################################################
########################################################################################################################.
    
class clonal_graph:
    
    '''
    A class that represents a clonal graph on amino acid sequences. 
    To create an instance, a shm_file and a corresponding seqs_file are needed. These files should be the output of IgEvolution. 
    '''
    
    def __init__(self, shm_file, seqs_file):
        self.shm_file = shm_file
        self.seqs_file = seqs_file
        self.simulations = []
        
        def get_original_sequences(a_seqs_file):
            '''
            Get original sequences from the seqs_file. 
            '''
            seqs = pd.read_csv(a_seqs_file, sep = '\t')
            seqs.Index = ['Node' + str(item) + 'a' for item in seqs.Index]
            org_sequences = seqs[['Index', 'AA_seq']].set_index('Index').to_dict()['AA_seq']
            return org_sequences
        
        def get_nxgraph_from_shm_file(shm_filename):
            '''
            Create a networkx graph from the shm_file.
            '''
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


        def get_pyvolve_phylogeny_from_nxgraph(G, root_seq):
            '''
            Transform the clonal graph into the format required by pyvolve. 
            '''
            tree_newick = networkx_to_newick(G)
            scale_tree = 1/len(root_seq)
            tree = pyvolve.read_tree(tree =tree_newick.replace('a', 'a:'+str(scale_tree)))
            return tree 
        
        self.org_seqs  = get_original_sequences(self.seqs_file)
        self.nx_graph, self.root_name, self.edges = get_nxgraph_from_shm_file(self.shm_file)
        self.root_sequence = self.org_seqs[self.root_name]
        self.pyvolve_phylogeny = get_pyvolve_phylogeny_from_nxgraph(self.nx_graph, self.root_sequence)
        
        
    def sims_to_null_dists_serially(self, a_delta_matrices_instance, return_mean_stats = False ):
        '''
        Uses the function 'get_test_stats' to generate test statistics from one simulation and then puts the test statistics from multiple simulations in one dataframe. 
        a_delta_matrices_instance is passed to the function 'get_test_stats'. 
        This function loops through all the simulations serially. Also see the function 'sims_to_null_dists_parallel'. 
        '''
        assert len(self.simulations) !=0, "No simulation data present in the clonal_graph object"
        test_stats_dfs = []
        for i, sim_seqs in enumerate(self.simulations):
            a_df = get_test_stats(sim_seqs, self.edges, a_delta_matrices_instance, return_mean_stats).add_suffix('_sim'+str(i))
            test_stats_dfs.append(a_df)

        return pd.concat(test_stats_dfs, axis = 1)
    
    
    def sims_to_null_dists_parallel(self, a_delta_matrices_instance, cpus, return_mean_stats = False ):
        '''
        Uses the function 'get_test_stats' to generate test statistics from one simulation and then puts the test statistics from multiple simulations in one dataframe. 
        a_delta_matrices_instance is passed to the function 'get_test_stats'. 
        This function processes multiple simulations at a time, using a different process for each.  
        '''
        assert len(self.simulations) !=0, "No simulation data present in the clonal_graph object"
        test_stats_dfs= Parallel(n_jobs=cpus)(delayed(get_test_stats)(sim_seqs, self.edges, a_delta_matrices_instance, return_mean_stats) \
                           for sim_seqs in self.simulations)
        return pd.concat(test_stats_dfs, axis = 1)

    

    def get_observed_stats(self, a_delta_matrices_instance, return_mean_stats = False ):
        '''
        Uses the function 'get_test_stats' on original sequences and returns a dataframe contatining positions that have non-zero test statistics along with their region (e.g.FR1).
        '''
        all_stats = get_test_stats(self.org_seqs, self.edges, a_delta_matrices_instance, return_mean_stats = return_mean_stats).transpose()
        non_zero_stats = all_stats[all_stats.apply(sum, 1) != 0]
        shm_df = pd.read_csv(self.shm_file, sep = '\t')
        pos_region = shm_df[['Position', 'Region',]].drop_duplicates().set_index('Position')
        non_zero_stats = pd.concat([non_zero_stats,pos_region], 1)
        non_zero_stats['Position'] = non_zero_stats.index
        return non_zero_stats.reset_index(drop = True).set_index(['Region', 'Position'])
    
    
    
def get_test_stats(simulated_seqs, edge_list, a_delta_matrices_instance, return_mean_stats ):
    '''
    Return the test statistic for all properties in a list of sequences connected with edges in the edge_list.
    For each pair of sequences connected by an edge, calculate the deltas for the 4 properties at all positions and get a dataframe. 
    Add all the dataframes element wise to get a site specific and property specific test statistic. 
    If return_mean_stats == True, return the mean of the non zero, instead of the sum over all edges. 
    '''
    
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
    '''
    Return the number of mutations at all positions in a list of sequences connected with edges in the edge_list.
    For each pair of sequences connected by an edge, see whether a mutation occured at all positions. 
    Sum over all edges element wise to get site specific number of mutations.  
    '''
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

########################################################################################################################
########################## Functions that use both Delta and clonal_graph ##############################################
########################################################################################################################

def plot_observed_stats(a_clonal_graph, a_delta_matrices_instance, CI_df, model,return_mean_stats= False, savefig = False, filename = None):
    property_colors = mpl.cm.get_cmap('tab10').colors
    mpl.rcParams.update({'font.size':14})
    
    # Get observed test statistics
    observed_stats = a_clonal_graph.get_observed_stats(a_delta_matrices_instance, return_mean_stats= return_mean_stats)
    observed_stats.sort_values([('Position')], inplace=True, ascending=False)
    
    # Set up plotting
    fig, axs = plt.subplots(2, 1, figsize =( 12, 8),gridspec_kw={'height_ratios': [8,1]}, sharex = True)
    
    # Subplot 1
    observed_stats[['volume', 'polarity', 'composition', 'aromaticity']].plot.barh(colors = property_colors, ax = axs[0])
    for i, key in enumerate(['volume', 'polarity', 'composition', 'aromaticity']):
        axs[0].axvline(CI_df[model][key], label = key, color = property_colors[i], linestyle = '--', )
    
    # Subplot 2
    pd.Series(CI_df[model]).plot.barh( color = 'slategrey', ax = axs[1])
    
    plt.subplots_adjust(hspace=0.05,)
    plt.tight_layout()
    if savefig: plt.savefig(filename, dpi = 300)
        

def create_output_df(a_clonal_graph, a_delta_matrices_instance, CI_df, model,return_mean_stats= False, save_to_file = False, filename = None):
    # Get observed test statistics
    observed_stats = a_clonal_graph.get_observed_stats(a_delta_matrices_instance,return_mean_stats= return_mean_stats)
    observed_stats.sort_values([('Position')], inplace=True, ascending=True)
    output = pd.concat([observed_stats, observed_stats.apply(lambda x: CI_df[model].add_suffix('_CI_max'), 1)], axis = 1)
    output.reset_index(inplace = True)
    if save_to_file: output.to_csv(filename, sep = '\t')
    return output

########################################################################################################################
########################################## Other utility functions ####################################################
########################################################################################################################


def CI_from_null(null_distbn_dict, level, plot= True, savefig = False, filename = None):
    Conf_int = pd.DataFrame()
    for model in null_distbn_dict:
        Conf_int[model] = null_distbn_dict[model].apply(lambda x : np.quantile(x, level), 1, )
    if plot:
        Conf_int.plot.barh(colormap = 'Accent', figsize = (12, 5));
        plt.title(f'{level*100}% confidence intervals');
        if savefig: plt.savefig(filename, dpi = 300)
    return Conf_int


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
    