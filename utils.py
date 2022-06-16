#!/usr/bin/env python
import gzip
import numpy as np
import pandas as pd
import os.path
import scipy.sparse
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
import sys

import scanpy as sc
import anndata

from genescreen_base_config import *

##### CREDIT ######
# most of these functions are Brian Hie's utility functions in Scanorama: https://github.com/brianhie/scanorama (bin/process.py)
###################


MIN_TRANSCRIPTS = 600

def load_tab(fname, max_genes=40000):
    if fname.endswith('.gz'):
        opener = gzip.open
    else:
        opener = open
        
    with opener(fname, 'r') as f:
        if fname.endswith('.gz'):
            header = f.readline().decode('utf-8').rstrip().split('\t')
        else:
            header = f.readline().rstrip().split('\t')
            
        cells = header[1:]
        X = np.zeros((len(cells), max_genes))
        genes = []
        for i, line in enumerate(f):
            if i > max_genes:
                break
            if fname.endswith('.gz'):
                line = line.decode('utf-8')
            fields = line.rstrip().split('\t')
            genes.append(fields[0])
            X[:, i] = [ float(f) for f in fields[1:] ]
    return X[:, range(len(genes))], np.array(cells), np.array(genes)



def load_mtx(dname):
    with open(dname + '/matrix.mtx', 'r') as f:
        while True:
            header = f.readline()
            if not header.startswith('%'):
                break
        header = header.rstrip().split()
        n_genes, n_cells = int(header[0]), int(header[1])

        data, i, j = [], [], []
        for line in f:
            fields = line.rstrip().split()
            data.append(float(fields[2]))
            i.append(int(fields[1])-1)
            j.append(int(fields[0])-1)
        X = csr_matrix((data, (i, j)), shape=(n_cells, n_genes))

    genes = []
    with open(dname + '/genes.tsv', 'r') as f:
        for line in f:
            fields = line.rstrip().split()
            genes.append(fields[1])
    assert(len(genes) == n_genes)

    return X, np.array(genes)



def load_h5(fname, genome='mm10'):
    try:
        import tables
    except ImportError:
        sys.stderr.write('Please install PyTables to read .h5 files: '
                         'https://www.pytables.org/usersguide/installation.html\n')
        exit(1)
    
    # Adapted from scanpy's read_10x_h5() method.
    with tables.open_file(str(fname), 'r') as f:
        try:
            dsets = {}
            for node in f.walk_nodes('/' + genome, 'Array'):
                dsets[node.name] = node.read()

            n_genes, n_cells = dsets['shape']
            data = dsets['data']
            if dsets['data'].dtype == np.dtype('int32'):
                data = dsets['data'].view('float32')
                data[:] = dsets['data']

            X = csr_matrix((data, dsets['indices'], dsets['indptr']),
                           shape=(n_cells, n_genes))
            genes = [ gene for gene in dsets['genes'].astype(str) ]
            assert(len(genes) == n_genes)
            assert(len(genes) == X.shape[1])
            
        except tables.NoSuchNodeError:
            raise Exception('Genome %s does not exist in this file.' % genome)
        except KeyError:
            raise Exception('File is missing one or more required datasets.')    

    return X, np.array(genes)



def process_tab(fname, min_trans=MIN_TRANSCRIPTS):
    X, cells, genes = load_tab(fname)

    gt_idx = [ i for i, s in enumerate(np.sum(X != 0, axis=1))
               if s >= min_trans ]
    X = X[gt_idx, :]
    cells = cells[gt_idx]
    if len(gt_idx) == 0:
        print('Warning: 0 cells passed QC in {}'.format(fname))
    if fname.endswith('.txt'):
        cache_prefix = '.'.join(fname.split('.')[:-1])
    elif fname.endswith('.txt.gz'):
        cache_prefix = '.'.join(fname.split('.')[:-2])
    elif fname.endswith('.tsv'):
        cache_prefix = '.'.join(fname.split('.')[:-1])
    elif fname.endswith('.tsv.gz'):
        cache_prefix = '.'.join(fname.split('.')[:-2])
    else:
        sys.stderr.write('Tab files should end with ".txt" or ".tsv"\n')
        exit(1)
        
    cache_fname = cache_prefix + '.npz'
    np.savez(cache_fname, X=X, genes=genes)

    return X, cells, genes

def process_mtx(dname, min_trans=MIN_TRANSCRIPTS):
    X, genes = load_mtx(dname)

    gt_idx = [ i for i, s in enumerate(np.sum(X != 0, axis=1))
               if s >= min_trans ]
    X = X[gt_idx, :]
    if len(gt_idx) == 0:
        print('Warning: 0 cells passed QC in {}'.format(dname))
    
    cache_fname = dname + '/tab.npz'
    scipy.sparse.save_npz(cache_fname, X, compressed=False)

    with open(dname + '/tab.genes.txt', 'w') as of:
        of.write('\n'.join(genes) + '\n')

    return X, genes



def process_h5(fname, min_trans=MIN_TRANSCRIPTS):
    X, genes = load_h5(fname)

    gt_idx = [ i for i, s in enumerate(np.sum(X != 0, axis=1))
               if s >= min_trans ]
    X = X[gt_idx, :]
    if len(gt_idx) == 0:
        print('Warning: 0 cells passed QC in {}'.format(fname))
    
    if fname.endswith('.h5'):
        cache_prefix = '.'.join(fname.split('.')[:-1])
        
    cache_fname = cache_prefix + '.h5.npz'
    scipy.sparse.save_npz(cache_fname, X, compressed=False)

    with open(cache_prefix + '.h5.genes.txt', 'w') as of:
        of.write('\n'.join(genes) + '\n')

    return X, genes


def load_data(name):
    if os.path.isdir(name):
        X, genes = process_mtx(name)
    elif os.path.isfile(name + '.h5.npz'):
        X = scipy.sparse.load_npz(name + '.h5.npz')
        with open(name + '.h5.genes.txt') as f:
            genes = np.array(f.read().rstrip().split())
    elif os.path.isfile(name + '.npz'):
        data = np.load(name + '.npz')
        X = data['X']
        genes = data['genes']
        data.close()
    elif os.path.isfile(name + '/tab.npz'):
        X = scipy.sparse.load_npz(name + '/tab.npz')
        with open(name + '/tab.genes.txt') as f:
            genes = np.array(f.read().rstrip().split())
    else:
        sys.stderr.write('Could not find: {}\n'.format(name))
        exit(1)
    genes = np.array([ gene.upper() for gene in genes ])
    return X, genes


def load_metadata(data_names):
    metadatasets = []
    for name in data_names:
        assert os.path.isdir(name)
        metadatasets.append( pd.read_csv(name + "/metadata.csv"))

    for i in range(1,len(metadatasets)):
        assert  metadatasets[i].columns.tolist() == metadatasets[0].columns.tolist()
        
    return metadatasets



def load_names(data_names, norm=True, log1p=False, verbose=True, get_metadata=False):
    # Load datasets.
    datasets = []
    genes_list = []
    n_cells = 0
    for name in data_names:
        X_i, genes_i = load_data(name)
        if norm:
            X_i = normalize(X_i, axis=1)
        if log1p:
            X_i = np.log1p(X_i)
        X_i = csr_matrix(X_i)
            
        datasets.append(X_i)
        genes_list.append(genes_i)
        n_cells += X_i.shape[0]
        if verbose:
            print('Loaded {} with {} genes and {} cells'.
                  format(name, X_i.shape[1], X_i.shape[0]))
    if verbose:
        print('Found {} cells among all datasets'
              .format(n_cells))

    if not get_metadata:
        return datasets, genes_list, n_cells
    else:
        metadatasets = load_metadata(data_names)
        return datasets, genes_list, n_cells, metadatasets



def save_datasets(datasets, genes, data_names, verbose=True,
                  truncate_neg=False):
    for i in range(len(datasets)):
        dataset = datasets[i].toarray()
        name = data_names[i]

        if truncate_neg:
            dataset[dataset < 0] = 0

        if os.path.isdir(name):
            outf = name + '/scanorama_corrected.txt'
        else:
            outf = name + '.scanorama_corrected.txt'
            
        with open(outf, 'w') as of:
            # Save header.
            of.write('Genes\t')
            of.write('\t'.join(
                [ 'cell' + str(cell) for cell in range(dataset.shape[0]) ]
            ) + '\n')

            for g in range(dataset.shape[1]):
                of.write(genes[g] + '\t')
                of.write('\t'.join(
                    [ str(expr) for expr in dataset[:, g] ]
                ) + '\n')



def save_anndata_datasets(datasets, metadatasets, genes, data_names, verbose=True,
                  truncate_neg=False):
    for i in range(len(datasets)):
        dataset = datasets[i].toarray()
        name = data_names[i]

        if truncate_neg:
            dataset[dataset < 0] = 0

        if os.path.isdir(name):
            outf = name + '/scanorama_corrected_adata.h5ad'
        else:
            outf = name + '.scanorama_corrected_adata.h5ad'

        metadata = metadatasets[i]
        gdata = pd.Series(genes).to_frame("gene").set_index("gene", drop=False)
        assert (dataset.shape[0] == metadata.shape[0] and dataset.shape[1] == gdata.shape[0])
        
        adata = anndata.AnnData(X = dataset, obs = metadata.set_index("cell", drop=False), var = gdata)
        adata.write(outf)


def save_as_single_anndata(outfile, datasets, metadatasets, genes, max_cell_count=10000000, truncate_neg=True, remove_outliers=True):
    adata_list = []
    for i in range(len(datasets)):
        dataset = datasets[i].toarray()

        if truncate_neg:
            dataset[dataset < 0] = 0

        metadata = metadatasets[i]
        gdata = pd.Series(genes).to_frame("gene").set_index("gene", drop=False)
        assert (dataset.shape[0] == metadata.shape[0] and dataset.shape[1] == gdata.shape[0])
        
        adata = anndata.AnnData(X = dataset, obs = metadata.set_index("cell", drop=False), var = gdata)
        adata_list.append(adata)

        
    adata_tot = anndata.AnnData.concatenate(*adata_list, join="inner")

    f1 = lambda a: (a.shape, a.var.shape, a.obs.shape)
    dbg_print ("Flag 56.20 ", f1(adata_tot), [f1(ax) for ax in adata_list])
    
    if adata_tot.shape[0] > max_cell_count:
        if remove_outliers:
            import geosketch, fbpca
            U, s, Vt = fbpca.pca(adata_tot.X, k=100)
            X_dimred = U[:, :100] * s[:100]
            n1 = int(0.1*adata_tot.shape[0])    
            sketch_index = geosketch.gs(X_dimred, n1, replace=False)
            a1 = np.full(adata_tot.shape[0], False).astype(bool) #np.full(False, adata_tot.shape[0])
            a1[sketch_index] =True
            adata_tot2 = adata_tot[~a1].copy()
            if adata_tot2.shape[0] >max_cell_count:
                sample_idxs = np.random.choice(adata_tot2.shape[0], max_cell_count, False)
                adata_tot3 = adata_tot2[sample_idxs,:].copy()
                adata_tot = adata_tot3
            else:
                adata_tot = adata_tot2
        else:
            sample_idxs = np.random.choice(adata_tot.shape[0], max_cell_count, False)
            adata_tot3 = adata_tot[sample_idxs,:].copy()
            adata_tot = adata_tot3
            
    # if adata_tot.shape[0] > max_cell_count:
    #     import geosketch, fbpca
    #     U, s, Vt = fbpca.pca(adata_tot.X, k=100)
    #     X_dimred = U[:, :100] * s[:100]
    #     sketch_index = geosketch.gs(X_dimred, max_cell_count, replace=False)
    #     adata_tot2 = adata_tot[sketch_index].copy()
    #     adata_tot = adata_tot2
        
    dbg_print("Flag 56.25 ", f1(adata_tot))
    adata_tot.write(outfile)

    
    
def BH_process(data_names, min_trans=MIN_TRANSCRIPTS):
    for name in data_names:
        if os.path.isdir(name):
            process_mtx(name, min_trans=min_trans)
        elif os.path.isfile(name) and name.endswith('.h5'):
            process_h5(name, min_trans=min_trans)
        elif os.path.isfile(name + '.h5'):
            process_h5(name + '.h5', min_trans=min_trans)
        elif os.path.isfile(name):
            process_tab(name, min_trans=min_trans)
        elif os.path.isfile(name + '.txt'):
            process_tab(name + '.txt', min_trans=min_trans)
        elif os.path.isfile(name + '.txt.gz'):
            process_tab(name + '.txt.gz', min_trans=min_trans)
        elif os.path.isfile(name + '.tsv'):
            process_tab(name + '.tsv', min_trans=min_trans)
        elif os.path.isfile(name + '.tsv.gz'):
            process_tab(name + '.tsv.gz', min_trans=min_trans)
        else:
            sys.stderr.write('Warning: Could not find {}\n'.format(name))
            continue
        print('Successfully processed {}'.format(name))

