
from helper import *
from preprocessing import SideInfo  # For processing data and side information
from embeddings_multi_task import Embeddings
from embeddings_weighted import W_Embeddings
from test_performance import cluster_test, HAC_getClusters
from utils import *
import os, argparse, pickle
from collections import defaultdict as ddict
from findkbyDB import findkbydb
from tqdm import tqdm





triples_list = []
true_ent2clust = ddict(set)
ckbdict = dict()
with open('data/myexp/ckbentid.txt', "r", encoding='utf-8') as f:
    for line in f:
        res = line.strip('\n').split('#####')
        qid = res[0]
        entnumber = res[1]
        ckbdict[qid] = entnumber
with open('data/myexp/active_links_gjq.txt', "r", encoding='utf-8') as rf:
    active_head_np_names = [line.strip('\n') for line in rf.readlines()]

count=0
print(len(active_head_np_names))
with open('data/myexp/cleanOKBtriple.txt', "r", encoding='utf-8') as f:
    for line in f:
        trp = {}
        res = line.strip('\n').split('#####')
        num = res[0]
        subnp = res[1]
        if subnp not in active_head_np_names:
            count+=1
            #print(type(subnp), type(active_head_np_names[0]))
            objnp = res[3]
            rel = res[2]
            subent = res[4]
            objent = res[5]
            trp['triple'] = [subnp, rel, objnp]
            trp['triple_unique'] = [subnp + '|' + num, rel + '|' + num,
                                    objnp + '|' + num]

            trp['true_sub_link'] = ckbdict[subent]
            trp['true_obj_link'] = ckbdict[objent]
            triples_list.append(trp)
            true_ent2clust[subnp + '|' + num].add(ckbdict[subent])


print(true_ent2clust)
true_clust2ent = invertDic(true_ent2clust, 'm2os')
print(len(true_clust2ent))