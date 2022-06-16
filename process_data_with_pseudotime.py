#!/usr/bin/env python

import pandas as pd
import numpy as np
import scipy, sklearn, os, sys, string, fileinput, glob, re, math, itertools, functools
import  copy, multiprocessing, traceback, logging, pickle, traceback
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


from genescreen_base_config import *


f_tfname = lambda s: s.replace("_extended","").lower()
    

def run_schema_ensemble(adata1, ptc, mincorrL =[0.1, 0.2, 0.3, 0.4, 0.5, 0.6], wL= [1000]):
    from schema import SchemaQP
    #from schema_qp import SchemaQP

    l = {}
    for mc in mincorrL:
        for w in wL:
            #sqp = SchemaQP(mc, w, params= {"dist_npairs": 1000000}, mode="scale")
            sqp = SchemaQP(mc, mode="scale", params= {"dist_npairs": 500000})
            try:
                prim = adata1.X if not scipy.sparse.issparse(adata1.X) else adata1.X.todense()
                sec =  np.ravel(adata1.obs[ptc].values)
                dbg_print("Flag 231.025 ", sec.shape, prim.shape, ptc, adata1.shape)
                dz1 = sqp.fit_transform(prim, [ sec ], [ 'numeric' ], [1])
                dbg_print("Flag 231.030 ", mc, w, dz1.shape, prim.shape, sec.shape, flush=True)
                ndim = prim.shape[1]
                wtsx = ndim*np.sqrt(np.maximum(sqp._wts/np.sum(sqp._wts), 0))
                l[(mc,w)] = wtsx
            except:
                print ("ERROR: schema failed for ", mc, w)
                #print ("Flag 232.032 ", np.mean(prim, 
                #raise
                continue
    return l



def convert_schemawts_to_df(adata1, l, norm_scheme = "0_1"):
    f_norm = {"0_1":  lambda v: (v-np.min(v))/(np.max(v)-np.min(v)), 
              "l2":   lambda v: v/np.sqrt(np.sum(v**2)),
              "none": lambda v: v,
              "rank": lambda v: scipy.stats.rankdata(v)/len(v),
              }

    x = []
    s = []
    for k,v in l.items():
        s.append( 'm{0}_w{1}'.format(k[0], k[1]))
        v1 = f_norm[norm_scheme](np.array(v))
        x.append(v1)
        
    x2 = pd.DataFrame(np.array(x))
    x2.index = s
    x2.columns = adata1.var.regulon
    return x2


def plotCellTypes(ax, adata1):
    pcmeans = adata1.obs.groupby("celltype")[["x","y"]].mean().reset_index()
    ax.scatter(adata1.obs.x, adata1.obs.y,s=1,c=pd.factorize(adata1.obs.celltype)[0]+1,alpha=0.4,cmap="jet")
    for i in range(pcmeans.shape[0]):
        ax.text(pcmeans["x"][i], pcmeans["y"][i], pcmeans["celltype"][i], size=12 )
    return




def get_regulon_location_plots(adata1, ptc, df, regulon_plot_cutoff_pctl):
    nfigs = 1+int(regulon_plot_cutoff_pctl*df.shape[0])
    df1 = df.sort_values('combined_rank', ascending=False).reset_index(drop=True)
    s = ['*'] + df1["regulon"].head(nfigs-1).values.tolist()
    
    
    ncols = 4
    nrows = math.ceil(nfigs/ncols)
    fig = plt.figure(figsize=(2.5*ncols,2.5*nrows))
    gs = fig.add_gridspec(nrows,ncols,wspace=0,hspace=0)
    for i,gene in enumerate(s):
        r,c = i//ncols, i%ncols
        #print(r,c)
        ax = fig.add_subplot(gs[r,c])
        if gene=="*":
            plotCellTypes(ax, adata1)
        else:
            try:
                j = list(adata1.var.regulon).index(gene)
            except:
                dbg_print("Flag 498.20 ", i, gene)
            jj = adata1.X[:,j] > 0.01
            xcol = adata1.obs["x"]
            ycol = adata1.obs["y"]
            ax.scatter(xcol, ycol,s=1,c="grey", alpha=0.30)
            ax.scatter(xcol[jj], ycol[jj],s=2,c="red")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(gene, pad =-12)
    fig.tight_layout()    
    return fig



def make_var_unique(v):
    s2cnt = defaultdict(int)
    vret = []
    for a in v:
        vret.append(a + ("" if s2cnt[a]==0 else ".{}".format(s2cnt[a])))
        s2cnt[a] += 1
    return vret


def generate_anndata_from_np_scdata(gexp_file, traj_file, regulon_file, tissue):
    d = pd.read_csv( gexp_file)
    d.columns = ["gene"] + list(d.columns)[1:]
    d1 = d.iloc[:,1:].T
    d1.columns = [str(c).upper() for c in d.gene]
    d1 = d1.reset_index().rename(columns={"index": "cell"})
    dbg_print("Flag 676.10 d1 ", d1.shape)

    ss = pd.read_csv( traj_file)
    for c in ss.columns:
        if c[:2]=="pt":
            ss.loc[~np.isfinite(ss[c]), c] = np.NaN
            
    dbg_print("Flag 676.20 ss ", ss.shape)

    regulons = pd.read_csv( regulon_file)
    dbg_print("Flag 676.30 regulons ", regulons.shape)
    
    assert "cell" in regulons.columns and "cell" in ss.columns and "cell" in d1.columns

    regulons.cell = make_var_unique(regulons.cell)
    ss.cell = make_var_unique(ss.cell)
    d1.cell = make_var_unique(d1.cell)
    
    common_cells = set(regulons.cell) & set(ss.cell) & set(d1.cell)
    dbg_print("Flag 676.33 common_cells ", len(common_cells))
    
    regulons = regulons[ regulons.cell.isin(common_cells)].sort_values("cell").reset_index(drop=True).set_index("cell")
    ss = ss[ ss.cell.isin(common_cells)].sort_values("cell").reset_index(drop=True).set_index("cell")
    d1 = d1[ d1.cell.isin(common_cells)].sort_values("cell").reset_index(drop=True).set_index("cell")
    if "celltype" not in ss.columns:
        if 'cell_type' in ss.columns:
            ss = ss.rename(columns = {'cell_type':'celltype'})
        else:
            ss["celltype"] = "UNK"

    dbg_print("Flag 676.40 ", d1.shape, ss.shape, regulons.shape)

    from anndata import AnnData
    import scanpy as sc
    
    adata_gexp = AnnData(X = d1)
    for c in ss.columns:
        adata_gexp.obs[c] = ss[c]
        
    adata_gexp.var_names_make_unique()
    sc.pp.normalize_total(adata_gexp)
    sc.pp.log1p(adata_gexp)    
    dbg_print("Flag 676.50 ", adata_gexp.shape, adata_gexp.obs.shape, adata_gexp.obs.columns)
    
    adata_regulons = AnnData(X = regulons)
    for c in ss.columns:
        adata_regulons.obs[c] = ss[c]
    adata_regulons.var["regulon"] = list(adata_regulons.var_names)
    
    dbg_print("Flag 676.60 ", adata_regulons.shape, adata_regulons.obs.shape, adata_regulons.obs.columns)

    return adata_regulons, adata_gexp



    
def generate_input_from_np_scdata(tissue, regulon_is_binarized, downsample_rate = 1.0):
    regulon_scoring = "binarized" if regulon_is_binarized else "continuous"
    
    if tissue == "gut-JL":
        ddir="~/work/perrimon-sc/data/2020-03-29_gut_scRNA-seq/"
    elif tissue == "blood-ST":
        ddir="~/work/perrimon-sc/data/2020-03-29_blood_scRNA-seq/"
    elif tissue == "KB":
        ddir="~/work/perrimon-sc/data/2020-05-20_KB_scRNA-seq/"
    elif tissue == "leukemia-ST":
        ddir="~/work/perrimon-sc/data/2020-08-14_leukemia_scRNA-seq/"
    elif tissue == "pijuansala-4samples":
        ddir="~/work/perrimon-sc/data/pijuansala-4samples/"
    elif tissue == "pijuansala-4samples-truncated":
        ddir="~/work/perrimon-sc/data/pijuansala-4samples-NOectoderm-NOmesenchyme-NOx-under-minus3/"
    elif tissue[:6] in ["pjs1_E", "pjs2_E", "pjs3_E"]:
        ddir="~/work/perrimon-sc/data/{}/".format(tissue.strip())
    elif tissue[:8] == 'bottcher':
        ddir="~/work/perrimon-sc/data/{}/".format(tissue.strip())
    elif tissue == 'humanblood-ST':
        ddir="~/work/perrimon-sc/data/2021-11-01-human-blood-lineages/"
    elif tissue == 'desplan_medulla1_all':
        ddir="~/work/perrimon-sc/data/desplan_medulla1_all/"
    else:
        assert 1==0
        
    gexp_file = ddir + "gexp.csv"
    if regulon_is_binarized:
        regulon_file = ddir + "binarized_regulonXcell_merged.csv"
    else:
        regulon_file = ddir + "continuous_regulonXcell_merged.csv"
        
    traj_file = ddir + "trajXcell.csv"

    dbg_print("Flag 679.01 ", tissue, ddir, gexp_file, traj_file, regulon_file)
    
    adata_regulon, adata_gexp = generate_anndata_from_np_scdata(gexp_file, traj_file, regulon_file, tissue)

    dbg_print("Flag 679.10 ", adata_regulon.shape, adata_gexp.shape)

    selected_cells = None
    if downsample_rate < 0.999:
        from fbpca import pca
        U, s, Vt = pca(adata_gexp.X, k=100) # E.g., 100 PCs.
        X_dimred = U[:, :100] * s[:100]
        from geosketch import gs
        N = int( downsample_rate *adata_gexp.shape[0]) # Number of samples to obtain from the data set.
        sketch_index = gs(X_dimred, N, replace=False)
        selected_cells = list(adata_gexp.obs_names[sketch_index])
        dbg_print("Flag 679.30 selecting {0} cells of {1}".format( len(selected_cells), adata_gexp.shape[0]))
        
    if selected_cells is not None:
        idx = adata_gexp.obs_names.isin(selected_cells)
        adata3 = adata_regulon[idx, :]
        adata4 = adata_gexp
    else:
        adata3 = adata_regulon
        adata4 = adata_gexp
        
    dbg_print("Flag 679.50 ", adata3.shape, adata_regulon.shape, adata_gexp.shape)
    return adata3, adata4, ddir



def convert_regulon_pt_scores_to_prob(ptvals, regulon_vals, do_binarized=True, nbins=100):
    v1min, v1max = np.nanmin(ptvals), np.nanmax(ptvals)
    if do_binarized:
        v2 = ptvals[regulon_vals>0.5]
    else:
        n = len(ptvals)
        v2a = np.random.choice(n, 10*n, True, regulon_vals/np.sum(regulon_vals))
        v2 = ptvals[v2a]

    h= np.histogram(v2, bins=nbins, range=(v1min, v1max))[0]
    return h/np.sum(h)

    


def get_regulon_concentration_ranks_probdist(adatax, do_binarized, ptcol, distmode = "W1"):
    f1 = lambda v: (np.min(v), np.max(v))
    f2 = lambda v: v/np.sum(v)
    L = []
    for i,c in enumerate(adatax.var_names):
        v1 = adatax.obs[ptcol].values
        v2 = adatax.X[:,i]

        do_oldstyle=False
        if do_oldstyle:
            v1min, v1max = f1(v1)
            dfx = pd.DataFrame({"idx": range(100)})
            p = pd.Series(np.floor((v1-v1min)/(v1max-v1min)*100).astype(int)).value_counts().to_frame().reset_index().rename(columns={"index":"idx",0:"p"})
            if do_binarized:
                q = pd.Series(np.floor((v1[adatax.X[:,i]>0.5]-v1min)/(v1max-v1min)*100).astype(int)).value_counts().to_frame().reset_index().rename(columns={"index":"idx",0:"q"})
            else:
                n = adatax.shape[0]
                v2 = np.random.choice(n, 10*n, True, f2(v2)) #f2(np.exp(v2)))
                q = pd.Series(np.floor((v1[v2]-v1min)/(v1max-v1min)*100).astype(int)).value_counts().to_frame().reset_index().rename(columns={"index":"idx",0:"q"})

            dfx = pd.merge(dfx, p, how="left")
            dfx.p.fillna(0,inplace=True)
            dfx = pd.merge(dfx, q, how="left")
            dfx.q.fillna(0,inplace=True)
            dfx.p = dfx.p/dfx.p.sum()
            dfx.q = dfx.q/dfx.q.sum()
        else:        
            p = convert_regulon_pt_scores_to_prob(v1, np.full(len(v1),1), True)
            q = convert_regulon_pt_scores_to_prob(v1, np.ravel(v2), do_binarized)
            dfx = pd.DataFrame({"p":p, "q":q})
            dfx.p.fillna(0, inplace=True)
            dfx.q.fillna(0, inplace=True)
            if dfx.p.sum() < 1e-12:
                dfx.p = 1.0/dfx.shape[0]
            if dfx.q.sum() < 1e-12:
                dfx.q = 1.0/dfx.shape[0]

            
        #display(dfx[(dfx.p==0) & (dfx.q>0)])
        #print(c)
        if distmode=="kl":
            L.append((np.sum(scipy.special.rel_entr(dfx.q.values, dfx.p.values)),c))
        elif distmode=="W1":
            #W1 dist bugfix below
            #L.append((np.sum(scipy.stats.wasserstein_distance(dfx.q.values, dfx.p.values)),c))
            vx = np.linspace(0,1,num=dfx.shape[0])
            L.append((np.sum(scipy.stats.wasserstein_distance(vx, vx.copy(), dfx.q.values, dfx.p.values)),c))
        else:
            assert 1==0

    dz = pd.DataFrame({"regulon": [a for _,a in L], distmode: [a for a,_ in L]}).sort_values(distmode,ascending=False).reset_index(drop=True)
    return dz



def get_regulon_location_concentration_ranks_schema(adata1, ptc):
    l = run_schema_ensemble(adata1, ptc)
    dbg_print("Flag 194.10 ", len(l))
    
    df1 = convert_schemawts_to_df(adata1, l, "rank") #"l2"
    dbg_print("Flag 194.20 ", df1.shape, df1.columns, df1.index)
    
    df2 = df1.mean().sort_values(ascending=False).to_frame().rename(columns={0:"schema"}).reset_index()
    dbg_print("Flag 194.30 ", df2.shape, df2.columns, df2.index)
    #df1.to_csv(sys.stdout, index=False)
    dbg_print("Flag 194.31 ")
    #df2.sort_values("schema",ascending=False).to_csv(sys.stdout, index=False)
    return df2



def get_regulon_location_concentration_ranks(adata, ptc, do_binarized):
    adata1 = adata[ ~adata.obs[ptc].isnull(), :]
    dbg_print("Flag 194.01 ", adata.shape, adata1.shape, ptc, do_binarized)
    print(adata1.var.head())
    print(adata1.obs.head())

    df2 = get_regulon_location_concentration_ranks_schema(adata1, ptc)
    
    distmode = "W1"
    df3 = get_regulon_concentration_ranks_probdist(adata1, do_binarized, ptc, distmode)
    dbg_print("Flag 194.40 ", distmode, df3.shape, df3.columns, df3.index)
    #df3.sort_values(distmode,ascending=False).to_csv(sys.stdout, index=False)
    
    dfz = pd.merge(df2, df3)
    dfz["combined_rank"] = 0.5*( dfz["schema"].rank(pct=True) + dfz[distmode].rank(pct=True))
    dfz = dfz.sort_values("combined_rank", ascending=False).reset_index(drop=True)
    dbg_print("Flag 194.60 ", dfz.shape, dfz.columns)
    dfz.to_csv(sys.stdout, index=False)
    print(dfz.corr(method='spearman'))
    return dfz


def produce_gexp_csv_for_plotting(gexp_file, coord_file, style, outfile=None):
    adata1 = generate_anndata_from_np_scdata(gexp_file, coord_file, style)
    
    df = pd.DataFrame(adata1.X)
    df.columns = adata1.var_names
    df["cell"] = adata1.obs_names
    df = pd.merge(df, adata1.obs.reset_index().rename(columns={"index":"cell"})[["cell","x","y","celltype"]])

    dbg_print("Flag 676.50 df\n",df.iloc[:5,:5])
    
    df1 = df.drop('cell',axis=1)
    # commenting out gene ranking, so we get log1p values
    # df1.iloc[:,:-3] = (df1.iloc[:,:-3].rank(axis=0,pct=True).values)

    dbg_print("Flag 676.60 df1\n",df1.iloc[:5,:5])
    dbg_print("Flag 676.70 df1\n",df1.iloc[:5,-5:])
    
    if outfile is not None:
        if '.csv' in outfile: df1.to_csv(outfile, index=False)
        if '.h' in outfile: df1.to_hdf(outfile, key="df")
        
    return df1



def read_tf_pairs(tfpair_file):
    import pyreadr
    tfdf = [v for v in pyreadr.read_r(tfpair_file).values()][0].iloc[:,:2] 
    tfdf = tfdf[ tfdf.gene.isin(tfdf.TF)].reset_index(drop=True)
    tfw = set( (tfdf.iat[i,0].lower().replace("_extended",""), tfdf.iat[i,1].lower().replace("_extended",""))
               for i in range(tfdf.shape[0]) )
    dbg_print("Flag 570.20 ", len(tfw), ("tet","vnd") in tfw, ("tet","cad") in tfw)
    return tfw





def select_TFs_from_ranked_list(known_regulons, regulon_rank_df, regulon_conc_thresh):

    w1 = regulon_rank_df.copy().sort_values("combined_rank",ascending=False).reset_index(drop=True)
    #w1 = regulon_rank_df.copy().sort_values("schema",ascending=False).reset_index(drop=True)
    n1 = int(w1.shape[0]*(1-regulon_conc_thresh))
    
    dbg_print("Flag 409.30 ", set(w1["regulon"].head(n1).tolist()) - set(known_regulons))
    s1 = set([f_tfname(s) for s in w1["regulon"].head(n1).tolist()]) #select all regulons of x if  "x" or "x_extended" made the rank cut-off
    return list(a for a in known_regulons if f_tfname(a) in s1)






def  get_scenic_tf_pairs(adata, ddir, tfset):
    fname1 = os.path.expanduser(ddir + "/regulonXtargets.csv")

    dbg_print("Flag 940.05 ", fname1, glob.glob(fname1))
    if glob.glob(fname1):
        d1 = pd.read_csv(fname1, index_col=0)
        tfs = set(d1["TF"].values)
        dbg_print("Flag 940.10 ", len(tfs), tfs)
        d1 = d1.loc[d1["gene"].isin(tfs),:].reset_index(drop=True)
        dbg_print("Flag 940.12 \n", d1.head(2))
        tfpairs = set()
        for i in range(d1.shape[0]):
            t1 = f_tfname(d1.iat[i,0])
            t2 = f_tfname(d1.iat[i,1])
            #dbg_print("Flag 940.15 ", f_tfname(t1), f_tfname(t2), f_tfname(t1) in tfset, f_tfname(t2) in tfset)
            if f_tfname(t1) in tfset and f_tfname(t2) in tfset:
                tfpairs.add((t1,t2))
    else:
        tfpair_file = ddir + "int/2.5_regulonTargetsInfo.Rds"
        tfpairs1 = read_tf_pairs(tfpair_file)
        tfpairs = set([(t1,t2) for t1,t2 in tfpairs1 if (f_tfname(t1) in tfset and f_tfname(t2) in tfset)])
                
    dbg_print("Flag 940.20 ", len(tfpairs), sum(1*(f_tfname(b)=="vnd") for a,b in tfpairs))
    return tfpairs




def get_graph_tf_pairs(tf_graph_file, tfset):
    df1 = pd.read_csv(tf_graph_file)
    tfpairs = set([(t1,t2) for t1,t2 in zip(df1["TF1"].values, df1["TF2"].values) if (f_tfname(t1) in tfset and f_tfname(t2) in tfset)])
    
    return tfpairs




def get_regulon_avg_ptscore(regulon_list, adata, ptc, do_binarized):
    adata1 = adata[ ~adata.obs[ptc].isnull(), :]

    regulon2avg_pt_score = {}
    for r in regulon_list:
        i = adata1.var_names.tolist().index(r)
        v1 = np.ravel(adata1.X[:,i]) >1e-5
        avg_pt_score = np.mean(adata1.obs[ptc][v1])
        
        #if ptc=="pt3": dbg_print("Flag 602.20 ", i, adata1.var_names[i], describe(np.ravel(adata1.X[:,i])))
        #avg_pt_score = np.sum(adata1.X[:,i]*adata1.obs[ptc])/np.sum(adata1.X[:,i])
        
        regulon2avg_pt_score[r] = avg_pt_score
        
    return regulon2avg_pt_score


        

def get_tf_pairs_supported_by_data_meandist(regulon_list1, tfpairs, adata, ptc, do_binarized, dist_quantile_thresh, tf_ptime_frac):
    dbg_print("Flag 802.01 ", ptc, len(tfpairs), ("tet","vnd") in tfpairs)

    adata1 = adata[ ~adata.obs[ptc].isnull(), :]
    vX = np.nansum((adata.X.astype(float) > 1e-12).astype(int), axis=0)
    vX1 = np.nansum((adata1.X.astype(float) > 1e-12).astype(int), axis=0)

    dbg_print("Flag 802.015 ", adata.shape, adata1.shape, [(r, vX1[i], vX1[i]/vX[i]) for i,r in enumerate(adata1.var_names.tolist())])
    
    regulon_list = [r for i,r in enumerate(adata1.var_names.tolist()) if (r in regulon_list1 and
                                                                          vX1[i] > 0  and
                                                                          vX1[i]/vX[i] > tf_ptime_frac) ]
    
    dbg_print("Flag 802.02 ", len(regulon_list1), len(regulon_list), set(regulon_list1) - set(regulon_list))
    
    regulon2avg_pt_score = get_regulon_avg_ptscore(regulon_list, adata, ptc, do_binarized)
    for c,s in regulon2avg_pt_score.items():
        if f_tfname(c) in ['tet','vnd','cad','ham','foxk','dp']:
            dbg_print("Flag 802.03 ", ptc, c, f_tfname(c), s)

    n = len(regulon_list)
    l1 = []
    for i in range(n):
        for j in range(i+1,n):
            if f_tfname(regulon_list[i]) == f_tfname(regulon_list[j]):
                continue
            l1.append( abs(regulon2avg_pt_score[regulon_list[i]]  - regulon2avg_pt_score[regulon_list[j]]))
        
    ptdist_thresh = np.quantile(l1, dist_quantile_thresh)
    dbg_print("Flag 802.07 ", ptc, ptdist_thresh, len(regulon_list), list(regulon2avg_pt_score.items())[:5])
    
    #tmp = defaultdict(list)
    #for r,s in regulon2avg_pt_score.items(): tmp[f_tfname(r)].append(s)
    #ptdist_thresh = 0.75*scipy.stats.iqr([np.mean(a) for a in tmp.values()])    
    
    selected_tfpairs_info = {}
    for r1 in regulon_list:
        r1a = f_tfname(r1)
        for r2 in regulon_list:
            r2a = f_tfname(r2)
            if r1a==r2a or ((r1a,r2a) not in tfpairs) or (not np.isfinite(regulon2avg_pt_score[r1])) or (not np.isfinite(regulon2avg_pt_score[r2])):
                continue
            dist1 = regulon2avg_pt_score[r2] - regulon2avg_pt_score[r1]
            dbg_print("Flag 802.11 ", r1a, r2a, dist1, r1, r2, regulon2avg_pt_score[r1], regulon2avg_pt_score[r2])
            if dist1 > ptdist_thresh: #ptdist_iqr:
                selected_tfpairs_info[(r1a,r2a)] = (ptc, dist1, r1, r2, regulon2avg_pt_score[r1], regulon2avg_pt_score[r2])
                

    v1 = pd.DataFrame([(*a,*b) for a,b in selected_tfpairs_info.items()], columns="TF1,TF2,pt,dist,TF1_regulon,TF2_regulon,score1,score2".split(","))
    v1 = v1.sort_values("dist",ascending=False).reset_index(drop=True)
    dbg_print("Flag 802.90 \n", v1.to_csv(sys.stdout, index=False)) # "\n".join(str(a) for a in selected_tfpairs_info.items()))
    fig = plot_pseudotime_and_tf_pairs2(adata, ptc, v1)
    return v1, fig




def get_regulonpair_W1_dist(adata, ptc, r1, r2, do_binarized, signed=True):
    adata1 = adata[ ~adata.obs[ptc].isnull(), :]

    i1 = adata1.var_names.tolist().index(r1)
    i2 = adata1.var_names.tolist().index(r2)
    
    ptvals = adata1.obs[ptc].values
    p1 = convert_regulon_pt_scores_to_prob(ptvals, np.ravel(adata1.X[:,i1]), do_binarized)
    p2 = convert_regulon_pt_scores_to_prob(ptvals, np.ravel(adata1.X[:,i2]), do_binarized)

    p1_median_idx = np.argmax(np.cumsum(p1) >= 0.5)
    p2_median_idx = np.argmax(np.cumsum(p2) >= 0.5)

    #W1-dist bugfix below
    #dist1 =  scipy.stats.wasserstein_distance(p1,p2)
    vx1, vx2 = np.linspace(0,1,num=len(p1)),  np.linspace(0,1,num=len(p2))
    dist1 =  scipy.stats.wasserstein_distance(vx1, vx2, p1, p2)
        
    distsign= 1 if not signed else np.sign(p2_median_idx - p1_median_idx)
    if f_tfname(r2)=="vnd" and f_tfname(r1)=="tet":
        print ("Flag 804.30 ", ptc, dist1, distsign, p1_median_idx, p2_median_idx, describe(ptvals[adata1.X[:,i1]>0.001]), describe(ptvals[adata1.X[:,i2]>0.001]), pd.DataFrame({"tet": p1, "vnd": p2}))
        
    return dist1*distsign



    
def get_tf_pairs_supported_by_data_W1(regulon_list1, tfpairs, adata, ptc, do_binarized, dist_quantile_thresh, tf_ptime_frac):
    dbg_print("Flag 803.01 ", ptc, len(tfpairs), ("tet","vnd") in tfpairs)

    adata1 = adata[ ~adata.obs[ptc].isnull(), :]
    vX = np.nansum((adata.X.astype(float) > 1e-12).astype(int), axis=0)
    vX1 = np.nansum((adata1.X.astype(float) > 1e-12).astype(int), axis=0)

    dbg_print("Flag 802.015 ", adata.shape, adata1.shape, [(r, vX1[i], vX1[i]/vX[i]) for i,r in enumerate(adata1.var_names.tolist())])
    
    regulon_list = [r for i,r in enumerate(adata1.var_names.tolist()) if (r in regulon_list1 and
                                                                          vX1[i] > 0  and
                                                                          vX1[i]/vX[i] > tf_ptime_frac) ]
    

    #adata1 = adata[ ~adata.obs[ptc].isnull(), :]
    #regulon_list = [r for i,r in enumerate(adata1.var_names.tolist()) if (r in regulon_list1 and np.sum(adata1.X[:,i])>1e-5 and np.sum(adata1.X[:,i]>1e-5)/np.sum(adata.X[:,i]>1e-5) > 0.33) ]
        
    n = len(regulon_list)
    l1 = []
    for i in range(n):
        for j in range(i+1,n):
            if f_tfname(regulon_list[i]) == f_tfname(regulon_list[j]):
                continue
            l1.append( get_regulonpair_W1_dist(adata, ptc, regulon_list[i], regulon_list[j], do_binarized, signed=False))
        
    ptdist_thresh = np.quantile(l1, dist_quantile_thresh)


    selected_tfpairs_info = {}
    for r1 in regulon_list:
        r1a = f_tfname(r1)
        for r2 in regulon_list:
            r2a = f_tfname(r2)
            if r1a==r2a or ((r1a,r2a) not in tfpairs): # or (not np.isfinite(regulon2avg_pt_score[r1])) or (not np.isfinite(regulon2avg_pt_score[r2])):
                continue
            dist1 = get_regulonpair_W1_dist(adata, ptc, r1, r2, do_binarized, signed=True)
            if dist1 > ptdist_thresh:
                selected_tfpairs_info[(r1a,r2a)] = (ptc, dist1, r1, r2, dist1/ptdist_thresh)
            

    v1 = pd.DataFrame([(*a,*b) for a,b in selected_tfpairs_info.items()], columns="TF1,TF2,ptcol,dist,TF1_regulon,TF2_regulon,dist_by_iqr".split(","))
    v1 = v1.sort_values("dist",ascending=False).reset_index(drop=True)
    dbg_print("Flag 803.90 \n", v1.to_csv(sys.stdout, index=False))
    fig = plot_pseudotime_and_tf_pairs2(adata, ptc, v1)
    return v1, fig




def plot_pseudotime_and_tf_pairs2(adata0, ptc, df_tfpairs_info):
    adata = adata0.copy()
    
    fig = plt.figure(figsize=(9,6))
    ax = fig.gca()
    
    coords = adata.obs.loc[:,["x","y","celltype"]].copy()
    celltype_means = coords.groupby("celltype").mean().reset_index()

    n0 = adata.shape[0]
    pt_scores = adata.obs[ptc]
    okpt = ~(pt_scores.isnull())
    
    tf2idxvec = {}
    for i,r in enumerate(adata.var_names.tolist()):
        v2 = np.ravel(adata.X[:,i])
        v2[~okpt] = 0
        v1 = v2 > 1e-5

        c = f_tfname(r)
        if c not in tf2idxvec: tf2idxvec[c] = (np.full(n0, False), np.full(n0, 0))
        #we'll duplicate the wts for cells that occur in both "tfA" and "tfA_extended", but that's probably not bad
        tf2idxvec[c] = (tf2idxvec[c][0] + v1,  tf2idxvec[c][1] + v2) 

    tf2loc = {}
    for c, (idx, wt) in tf2idxvec.items():
        if np.sum(idx)<1:
            tf2loc[c] = (np.NaN, np.NaN)
        else:
            #i = adata.var_names.tolist().index(c)
            #v1 = np.ravel(adata.X[:,i])
            #tf2loc[c] = (np.average(coords.x[idx], weights=v1[idx]), np.average(coords.y[idx], weights=v1[idx]))
            tf2loc[c] = (np.average(coords.x[idx], weights=wt[idx]), np.average(coords.y[idx], weights=wt[idx]))
    
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='6%', pad=0.05)

    sc1 = ax.scatter(coords.x, coords.y, s=1, c='grey', alpha=0.3) #background of all points

    sc2 = ax.scatter(coords.x[okpt], coords.y[okpt], s=4, c=pt_scores[okpt], cmap="coolwarm") #this pt traj, in color
    cbar = fig.colorbar(sc2, cax=cax, orientation='vertical')
    cbar.ax.set_ylabel("Pseudotime value")

    xrng = coords.x.max() - coords.x.min()
    yrng = coords.y.max() - coords.y.min()

    all_texts = []
    for i in range(celltype_means.shape[0]):
        txt1 = ax.text(celltype_means["x"][i], celltype_means["y"][i], celltype_means["celltype"][i], size=12, color="red" )
        txt1.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='w')])
        all_texts.append(txt1)

    f_check_tf2loc_nan = lambda a: not (np.isfinite(tf2loc[a][0]) and np.isfinite(tf2loc[a][1]))
    
    tf_freq = defaultdict(int)
    for i, (tf1, tf2) in enumerate(zip(df_tfpairs_info["TF1"].values, df_tfpairs_info["TF2"].values)):
        if f_check_tf2loc_nan(tf1) or f_check_tf2loc_nan(tf2):
            continue

        x1, y1 = tf2loc[tf1][0], tf2loc[tf1][1]
        x2, y2 = tf2loc[tf2][0], tf2loc[tf2][1]
        
        color = 'black' 
        ax.arrow(x1, y1, x2-x1, y2-y1, width=0.005, color=color, head_width=0.2, head_length=0.25, overhang=0.6, length_includes_head=True)
        tf_freq[tf1] += 1
        tf_freq[tf2] += 1
            
    l = sorted([(v,k) for k,v in tf_freq.items()])

    txtx = {}
    prev_locs = []
    for _, tfstr in l:
        loc = tf2loc[tfstr]
        txtx[tfstr] = ax.text(loc[0], loc[1], tfstr, size=8, color="black" )
        txtx[tfstr].set_path_effects([PathEffects.withStroke(linewidth=4, foreground='w')])
        all_texts.append(txtx[tfstr])
        
    

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Pseudotime {}".format(ptc),  pad =-12)

    from adjustText import adjust_text
    adjust_text(all_texts, ax=ax, expand_text = (1.1, 1.25), lim=1000) #arrowprops=dict(arrowstyle="->", color='darkgreen', lw=0.5), 

    fig.tight_layout()
    return fig




if __name__ == "__main__":
    # sys.path.append(os.path.join(sys.path[0],'/afs/csail.mit.edu/u/r/rsingh/work/schema/schema'))
    # from schema_qp import SchemaQP
    # prim = np.random.uniform(0,1,400).reshape(40,10)
    # sec = np.random.uniform(0,1,40)
    # s = SchemaQP(0.9, mode='scale')
    # s.fit(prim, [sec], ['numeric'], [1])

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", help="which code path to run. see main(..) for details", type=str, choices = ["1_rank_TFs_by_concentration", "2_pair_TFs_by_ptime"])
    parser.add_argument("--tissue", help="which tissue data to analyze: gut-JL, blood-ST, KB, leukemia-ST", type=str)
    parser.add_argument("--outdir", help="output directory (can set to '.')", type=str, default="/afs/csail.mit.edu/u/r/rsingh/work/perrimon-sc/data/processed/")
    parser.add_argument("--outsfx", help="suffix to use when producing output files")
    parser.add_argument("--tf_graph", help="file containing TF1->TF2 graph as edge listing", type=str, default=PROJ_DIR + "/data/ext/dmel_TF_map_maxrank-100.csv")
    #parser.add_argument("--tf_graph", help="file containing TF1->TF2 graph as edge listing", type=str, default=PROJ_DIR + "/data/ext/mmus_TF_map_maxrank-100.csv")
    parser.add_argument("--extra", help="put this as the LAST option and arbitrary space-separated key=val pairs after that", type=str, nargs='*')


    args = parser.parse_args()
    assert args.mode is not None

    assert args.tissue[:6] in ["pjs1_E", "pjs2_E", "pjs3_E"] or args.tissue[:8] in ['bottcher'] or  (args.tissue in ["gut-JL", "blood-ST", "KB", "leukemia-ST", "pijuansala-4samples", "pijuansala-4samples-truncated", "humanblood-ST", "desplan_medulla1_all"])
    
    extra_args = dict([a.split("=") for a in args.extra]) if args.extra else {}
    
    pd.set_option('use_inf_as_na', True)

    do_binarized = extra_args.get("regulon_scoring","binarized") == "binarized"
    adata, adata_gexp, ddir = generate_input_from_np_scdata(args.tissue, do_binarized, downsample_rate = float(extra_args.get("sample_rate",1.0)))

    if args.mode == "1_rank_TFs_by_concentration":

        if do_binarized:
            adata_bin = adata
        else:
            adata_bin, _, _ = generate_input_from_np_scdata(args.tissue, 1, 1)

        ptcols = [ c for c in adata.obs.columns if c[:2]=="pt"]
        for ptc in ptcols:
            s2 = "ptimefit_{}_{}".format(ptc, args.outsfx)
            
            df = get_regulon_location_concentration_ranks(adata, ptc, do_binarized)
            df.to_csv("{0}/regulonloc_dfrank_{1}.csv".format(args.outdir, s2), index=False)
            
            fig = get_regulon_location_plots(adata_bin, ptc, df, float(extra_args.get("regulon_plot_cutoff_pctl",0.2)))
            fig.savefig("{0}/regulonloc_plots_{1}.png".format(args.outdir, s2))
            

            
    elif args.mode == "2_pair_TFs_by_ptime":
        dist_quantile_thresh = float(extra_args.get("dist_quantile_thresh",0.8))
        tf_ptime_frac = float(extra_args.get("tf_ptime_frac",0.001))
        
        ptcols = [ c for c in adata.obs.columns if c[:2]=="pt"]
        for ptc in ptcols:
            s2 = "ptimefit_{}_{}".format(ptc, args.outsfx)

            if "regulonloc_dfrank_filepat" in extra_args:
                a = extra_args["regulonloc_dfrank_filepat"]
                apat = "{}/regulonloc_dfrank_ptimefit_{}*{}*.csv".format(args.outdir, ptc, a)
            else:
                apat = "{}/regulonloc_dfrank_{}*.csv".format(args.outdir, s2)
                
            dbg_print("Flag 563.46 ", describe(np.nansum(adata.X,axis=0)))
            
            la = glob.glob(apat)
            assert len(la)>0, "Couldn't find regulonloc_dfrank file [{}]".format(apat)
            fname1 = la[0]                

            regulon_rank_df = pd.read_csv(fname1) #"{0}/{1}}.csv".format(args.outdir, s3))
            regulon_list = select_TFs_from_ranked_list(adata.var_names.tolist(), regulon_rank_df, float(extra_args.get("regulon_conc_thresh",0.33)))

            dbg_print("Flag 563.47 ", describe(np.nansum(adata.X,axis=0)))
            
            tfset = set([f_tfname(c) for c in regulon_list])
            dbg_print("Flag 563.52 ", regulon_list, tfset)

            
            if int(extra_args.get("do_only_scenic_tf_edges",0))>0.5:
                tfpairs = get_scenic_tf_pairs(adata, ddir, tfset)
            else:
                tfpairs = get_graph_tf_pairs(args.tf_graph, tfset)
                
            dbg_print("Flag 563.525 ", describe(np.nansum(adata.X,axis=0)))
            
            dbg_print("Flag 563.55 {} {}".format(ptc, tfpairs))
            if int(extra_args.get("allow_flipped_tf_pairing",0)) > 0.5:
                a = copy.copy(tfpairs)
                for t1,t2 in a:
                    if (t2,t1) not in  a:
                        tfpairs.add((t2,t1))

            if extra_args.get("tf_dist_style", "W1").lower() == "meandist":
                selected_tfpairs_info, fig = get_tf_pairs_supported_by_data_meandist(regulon_list, tfpairs, adata, ptc, do_binarized, dist_quantile_thresh, tf_ptime_frac)
            else:
                selected_tfpairs_info, fig = get_tf_pairs_supported_by_data_W1(regulon_list, tfpairs, adata, ptc, do_binarized, dist_quantile_thresh, tf_ptime_frac)

            dbg_print("Flag 563.57 ", describe(np.nansum(adata.X,axis=0)))
            
            selected_tfpairs_info.to_csv("{0}/matched_tfpairs_info_{1}.txt".format(args.outdir, s2), index=False)
            fig.savefig("{0}/matched_tfpairs_{1}.png".format(args.outdir, s2))

        
