#!/usr/bin/env python

import warnings
warnings.simplefilter("ignore")

import pandas as pd
import numpy as np
import scipy, sklearn, os, sys, string, fileinput, glob, re, math, itertools, functools
import  copy, multiprocessing, traceback, logging, pickle, traceback, tempfile, csv
import scipy.stats, sklearn.decomposition, sklearn.preprocessing, sklearn.covariance
from scipy.stats import describe
from scipy import sparse
import os.path
import scipy.sparse
from scipy.sparse import csr_matrix, csc_matrix
from sklearn.preprocessing import normalize
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.patheffects as PathEffects
    
import anndata
import scanpy as sc

from haystack_base_config import *
import utils
import preprocess_dataset


    
BASE_DATADIR="/afs/csail.mit.edu/u/r/rsingh/work/perrimon-sc/data/ext/desplan_fly_brain_nature/medulla-sketched-10k/"
DESPLAN_OUTDIR_PFX="desplan_medulla1"



def save_desplan_adata_with_pseudotime():
    adata = sc.read(BASE_DATADIR + "/GSE167266_OL.L3_P15_merged_RS-Medulla-only-sketched-10k.h5ad")
    adata.obs['gcnt'] = adata.obs['nFeature_RNA']
    adata.obs['likely_progenitor'] = 1*(adata.obs['Nikos_Neset_ID'].isin(['L3_6','L3_9']))

    # this is from jupyter analysis of desplan_fly_brain_v1
    
    iroot_cells = ((adata.obs['likely_progenitor'] > 0.5) & #Nikos_Neset_ID'].isin(['L3_6','L3_9'])) & 
             (adata.obs['gcnt'] > 2500))

    iroot_center = adata.obsm['X_umap'][iroot_cells].mean(axis=0)
    iroot_dist = ((adata.obsm['X_umap']-iroot_center[None,:])**2).sum(axis=1)
    adata.obs['iroot_dist'] = np.where(iroot_cells, iroot_dist, 1e9)

    iroot = np.argmin(adata.obs['iroot_dist'])
    adata.uns['iroot'] = iroot

    adata2 = anndata.AnnData(X = np.expm1(adata.raw.X.todense())) #adata.raw.X.todense()) 
    adata2.var_names=adata.raw.var_names
    adata2.obs_names=adata.obs_names
    adata2.obs = adata.obs
    adata2.uns['iroot'] = iroot
    adata2.obsm['X_umap'] = adata.obsm['X_umap']

    sc.pp.neighbors(adata2, use_rep='X_umap') #not using X_UMAP led to worse dpt
    sc.tl.diffmap(adata2)
    sc.tl.dpt(adata2, n_dcs=10, n_branchings=0)

    def rescale_dpt(v):
        vmin, vmax = np.nanquantile(v,[0.025,0.975])
        v2 = 10*np.minimum(1.0, np.maximum(0.0, (v-vmin)/(vmax-vmin)))
        return v2

    v = adata2.obs['dpt_pseudotime']
    adata2.obs['dpt_pseudotime'] = rescale_dpt(np.where( np.isfinite(v), v, np.NaN))
    v2 = 1*np.ravel((adata2.obsm['X_umap'][:,0] + adata2.obsm['X_umap'][:,1]) > 12)
    adata2.obs['dpt_pseudotime'] = np.where( v2 & (adata2.obs['likely_progenitor']<0.5), np.NaN, adata2.obs['dpt_pseudotime'])

    dbg_print("Flag 702.50 ", np.quantile(adata2.obs['dpt_pseudotime'].tolist(), 0.1*np.arange(10)))

    dbg_print("Flag 702.60 ", adata2.shape, np.quantile(adata2.obs['dpt_pseudotime'].tolist(), 0.1*np.arange(10)))
    adata2.write("{}/adata_subset_all.h5ad".format(BASE_DATADIR))


def run_scenic_in_docker(stages, num_jobs=24, HVG=40000):
    pass #this was done by hand. Look at preprocess_pijuansala_data or preprocess_bottcher for better examples


    
def binarize_scenic_matrix(sstr):
    ddir = BASE_DATADIR

    d0 = pd.read_csv("{}/auc_mtx.csv".format(ddir))
    d = d0.set_index("Cell")
    dbg_print("Flag 803.20 ", sstr, d.shape)
    
    import pyscenic.binarization
    d1, _  = pyscenic.binarization.binarize(d)
    dbg_print("Flag 803.30 ", d1.shape)
    d1.to_csv("{}/auc_binarized.csv".format(ddir), index=True, index_label="cell")

    
    
def write_multistage_cell_metadata(sstr):
    pdir = BASE_DATADIR

    dbg_print("Flag 720.10 ", sstr)
    import scanpy as sc
    adata_file = '{}/adata_subset_{}.h5ad'.format(BASE_DATADIR, sstr)
    adata = sc.read(adata_file)
    exprmat_file = '{}/expr_mat.tsv'.format(pdir)

    exprmat_cells = [a.split('\t')[0] for a in open(exprmat_file,'r')][1:]
    selected_cells = [a for a in exprmat_cells if a in adata.obs_names]
    dbg_print("Flag 720.20 ", adata.shape, len(selected_cells))
    
    df = adata.obs[ adata.obs.index.isin(selected_cells)] #.copy().reset_index(drop=True)
    df['cell'] = df.index.tolist()
    dbg_print("flag 720.40 ", df.shape)
    assert list(df.index) == selected_cells
    
    df.to_csv('{}/metadata.csv'.format(pdir), index=False)




def prepare_genescreen_data(sstr):
    ddir = BASE_DATADIR
    outdir = "/data/cb/rsingh/work/perrimon-sc/data/{}_{}".format(DESPLAN_OUTDIR_PFX, sstr)
    if not os.path.isdir(outdir):
        os.system("mkdir -p {}".format(outdir))
        
    dbg_print("Flag 552.10 ", sstr, ddir, outdir)

    import scanpy as sc
    adata_file = '{}/adata_subset_{}.h5ad'.format(BASE_DATADIR, sstr)
    adata = sc.read(adata_file)
    selected_cells = adata.obs_names.tolist()
    
    d1o = pd.read_csv("{}/expr_mat.tsv".format(ddir), sep="\t")
    d1a = d1o.loc[ selected_cells,:]
    dbg_print("Flag 552.15 ", d1o.shape, d1a.shape)
    
    d1b = d1a.T
    d1b.columns = [s.split('-')[0] for s in d1b.columns]
    dbg_print("Flag 552.20 ", d1b.shape, d1b.index[:5], d1b.columns[:5])
    d1b.to_csv("{}/gexp.csv".format(outdir), index=True, index_label="")

    
    d2a = pd.read_csv("{}/auc_mtx.csv".format(ddir))
    d2o = d2a.set_index("Cell")
    d2b = d2o.loc[selected_cells,:]
    dbg_print("Flag 552.25 ", d2a.shape, d2o.shape, d2b.shape)
    
    d2b.index = [s.split("-")[0] for s in d2b.index]
    d2b.columns = [s.replace('(+)','') for s in d2b.columns]
    dbg_print("Flag 552.30 ", d2b.shape, d2b.index[:5], d2b.columns[:5])
    d2b.to_csv("{}/continuous_regulonXcell_merged.csv".format(outdir), index=True, index_label="cell")

    
    d3a = pd.read_csv("{}/auc_binarized.csv".format(ddir))
    d3o = d3a.set_index("cell") #yeah this is lower case accidentally
    d3b = d3o.loc[selected_cells,:]
    dbg_print("Flag 552.35 ", d3a.shape, d3o.shape, d3b.shape)
    
    d3b.index = [s.split("-")[0] for s in d3b.index]
    d3b.columns = [s.replace('(+)','') for s in d3b.columns]
    dbg_print("Flag 552.40 ", d3b.shape, d3b.index[:5], d3b.columns[:5])
    d3b.to_csv("{}/binarized_regulonXcell_merged.csv".format(outdir), index=True, index_label="cell")

    
    tf2tgts = defaultdict(set)
    rex1 = re.compile(r"\('(.*?)',")
    for i,r in enumerate(csv.reader(open("{}/regulons.csv".format(ddir), "r"))):
        if i<3 or len(r) <10: continue
        tf = r[0]
        tf2tgts[tf].update(  [m.group(1) for m in re.finditer(rex1, r[9])] )

    L = []
    for t in tf2tgts:
        for v in tf2tgts[t]:
            L.append([t,v])
            
    d4 = pd.DataFrame(L, columns=["TF","gene"])
    dbg_print("Flag 552.50 ", d4.shape, len(L), len(tf2tgts), d4.head(5))
    d4.to_csv("{}/regulonXtargets.csv".format(outdir), index=True)


    d5a = pd.read_csv("{}/metadata.csv".format(ddir))
    d5a = d5a.set_index('cell', drop=False)
    dbg_print("Flag 552.60 ", d5a.shape)
    d5a.to_csv('{}/orig_data_auxinfo.csv'.format(outdir), index=True, index_label='')


    adata_file = '{}/adata_subset_{}.h5ad'.format(BASE_DATADIR, sstr)
    adata = sc.read(adata_file)
    dtraj = adata.obs
    dtraj.index.name = 'cell'
    dtraj = dtraj.reset_index()
    dtraj['cell'] = dtraj['cell'].apply(lambda s: s.split('-')[0])
    dtraj["pt1"] = dtraj["dpt_pseudotime"]
    dtraj['x'] = np.ravel(adata.obsm['X_umap'][:,0])
    dtraj['y'] = np.ravel(adata.obsm['X_umap'][:,1])
    dtraj.to_csv("{}/trajXcell.csv".format(outdir), index=False)
    #os.system("cp {}/trajXcell.csv {}/".format(ddir, outdir))
    dbg_print("Flag 552.70 ")

    

#########################################################

if __name__ == "__main__":
    celltypes_list = ["all"]
    job_stages = [6] #[2,4,5,6] 

   
    if len(sys.argv)>1:
        celltypes_list = sys.argv[1].split(':')  
 

    if 1 in job_stages:
        pass # this is for data download, already downloaded here
        
    if 2 in job_stages:
        for s in celltypes_list:
            dbg_print("Flag 945.10 ", s)
            save_desplan_adata_with_pseudotime()
        
    if 3 in job_stages:
        pass # this was done by hand
        # for s in celltypes_list:
        #     dbg_print("Flag 945.20 ", s)
        #     run_scenic_in_docker(s, 24)

        
    if 4 in job_stages:
        for s in celltypes_list:
            dbg_print("Flag 945.30 ", s)
            binarize_scenic_matrix(s) # ---> this needs to be run on imp37 (virtual env), which has pyscenic
            dbg_print("Flag 945.33 ", s)
            write_multistage_cell_metadata(s)

    if 5 in job_stages:
        # this is for monocle, have already pseudotime
        pass

    if 6 in job_stages:
        for s in celltypes_list:
            dbg_print("Flag 945.50 ", s)
            prepare_genescreen_data(s)
