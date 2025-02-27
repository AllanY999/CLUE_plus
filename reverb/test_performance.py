from helper import *
from utils import *
from metrics import evaluate  # Evaluation metrics
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
import math
import pickle
from scipy import sparse
from fast_pagerank import pagerank
ave = True


def HAC_getClusters(params, embed, cluster_threshold_real,threshold_or_cluster):

    embed_dim = 300
    dist = pdist(embed, metric=params.metric)

    clust_res = linkage(dist, method=params.linkage)
    if threshold_or_cluster == 'threshold':
        labels = fcluster(clust_res, t=cluster_threshold_real, criterion='distance') - 1
    else:
        labels = fcluster(clust_res, t=cluster_threshold_real, criterion='maxclust') - 1


    clusters = [[] for i in range(max(labels) + 1)]
    for i in range(len(labels)):
        clusters[labels[i]].append(i)

    clusters_center = np.zeros((len(clusters), embed_dim), np.float32)
    for i in range(len(clusters)):
        cluster = clusters[i]
        if ave:
            clusters_center_embed = np.zeros(embed_dim, np.float32)
            for j in cluster:
                embed_ = embed[j]
                clusters_center_embed += embed_
            clusters_center_embed_ = clusters_center_embed / len(cluster)
            clusters_center[i, :] = clusters_center_embed_
        else:
            sim_matrix = np.empty((len(cluster), len(cluster)), np.float32)
            for i in range(len(cluster)):
                for j in range(len(cluster)):
                    if i == j:
                        sim_matrix[i, j] = 1
                    else:
                        if params.metric == 'cosine':
                            sim = cos_sim(embed[i], embed[j])
                        else:
                            sim = np.linalg.norm(embed[i] - embed[j])
                        sim_matrix[i, j] = sim
                        sim_matrix[j, i] = sim
            sim_sum = sim_matrix.sum(axis=1)
            max_num = cluster[int(np.argmax(sim_sum))]
            clusters_center[i, :] = embed[max_num]
    # print('clusters_center:', type(clusters_center), clusters_center.shape)
    return labels, clusters_center


def generatelinkseed(myclust2ent,ent2id,id2ent,ite):
    print('generate soft link seed for iter'+str(ite+1))
    resdict = {}
    fname4 = 'data/myexp/numbertrueDict.txt'
    file4 = open(fname4, 'r', encoding='utf-8')
    for line in file4:
        dict1 = eval(line)
        resdict.update(dict1)
    def negexp(num):
        return math.exp(-num)

    def entropy(dic):
        sum = 0
        for ent, count in dic.items():
            sum += count
        result = 0
        for ent, count in dic.items():
            p = count / sum
            result -= p * math.log(p)
        return result
    dst = open('data/myexp/softlinkseed.txt', "w", encoding='utf-8')
    result = pickle.load(open('iter'+str(ite), 'rb'))
    confidence=pickle.load(open('iter'+str(ite)+'confidence', 'rb'))
    count=dict(dict())

    clusterentropy={}
    maxcount={}
    clusterconfidence={}
    errorcount=0
    for label in myclust2ent.keys():
        count[label]={}
        clust=myclust2ent[label]
        for np in clust:
            if np in result:
                link=result[np]
                if link in count[label]:
                    count[label][link]+=confidence[np]
                else:
                    count[label][link] =confidence[np]

            else:
                errorcount+=1
        if len(count[label])>0:
            clusterentropy[label]=entropy(count[label])
            maxcount[label]=max(count[label],key=lambda k:count[label][k])
            #clusterconfidence[label]=negexp(entropy(count[label]))

    labels=[]
    error=0
    for label,v in clusterentropy.items():
        if math.log(2-math.exp(-v),math.e)<math.exp(-1/(ite+1)) :
            labels.append(label)
            clusterconfidence[label] = 1 -(math.log(2-math.exp(-v),math.e)/math.exp(-1/(ite+1)))
    for label in labels:
        for np in myclust2ent[label]:
            if np not in resdict.keys():
                dst.writelines(np+'#####'+maxcount[label]+'#####'+str(clusterconfidence[label])+'\n')
            else:
                if maxcount[label] in resdict[np]:
                    dst.writelines(np + '#####' + maxcount[label] +'#####'+str(clusterconfidence[label])+'\n')
                else:
                    error+=1



############################## gjq ########################################################
    with open("data/myexp/gjq_mid_res_" + str(ite) + ".pkl", "wb") as file:
        pickle.dump({
            'myclust2ent': myclust2ent,
            'maxcount': maxcount,
            'clusterentropy': clusterentropy,
            'ite': ite
        }, file)

    error=0
    from collections import defaultdict
    from itertools import combinations
    ##### 3 #####
    ## influnce ##
    # if os.path.exists('data/myexp/activate_score3_gjq.pkl'):
    #     activate_score3 = pickle.load(open('data/myexp/activate_score3_gjq.pkl', 'rb'))
    # else:
    #     connected_nps, activate_score3 = defaultdict(set), {}
    #     with open('data/myexp/cleanOKBtriple.txt', "r", encoding='utf-8') as f:
    #         for line in f.readlines():
    #             np1, np2 = line.strip('\n').split('#####')[1], line.strip('\n').split('#####')[3]
    #             connected_nps[np1].add(np2)
    #             connected_nps[np2].add(np1)
    #         for np, c_nps in connected_nps.items():
    #             activate_score3[np] = sum([1 / len(connected_nps[c_np]) for c_np in c_nps])
    #     pickle.dump(activate_score3, open('data/myexp/activate_score3_gjq.pkl', 'wb'))
    ############
    # degree ##
    if os.path.exists('data/myexp/activate_score_degree_gjq.pkl'):
        activate_score_degree = pickle.load(open('data/myexp/activate_score_degree_gjq.pkl', 'rb'))
    else:
        connected_nps, activate_score_degree = defaultdict(set), {}
        with open('data/myexp/cleanOKBtriple.txt', "r", encoding='utf-8') as f:
            for line in f.readlines():
                np1, np2 = line.strip('\n').split('#####')[1], line.strip('\n').split('#####')[3]
                connected_nps[np1].add(np2)
                connected_nps[np2].add(np1)
            for np, c_nps in connected_nps.items():
                activate_score_degree[np] = len(c_nps) # sum([1 / len(connected_nps[c_np]) for c_np in c_nps])
        pickle.dump(activate_score_degree, open('data/myexp/activate_score_degree_gjq.pkl', 'wb'))
    ###########
    ## pagerank ##
    # if os.path.exists('data/myexp/activate_score_page_gjq.pkl'):
    #     activate_score_page = pickle.load(open('data/myexp/activate_score_page_gjq.pkl', 'rb'))
    # else:
    #     # pip install fast_pagerank
    #
    #     import numpy as np
    #     all_edges, max_np = [], -1000000
    #     with open('data/myexp/cleanOKBtriple.txt', "r", encoding='utf-8') as f:
    #         for line in f.readlines():
    #             np1, np2 = int(line.strip('\n').split('#####')[1]), int(line.strip('\n').split('#####')[3])
    #             max_np = max(max_np, max(np1, np2))
    #             all_edges.append([np1, np2])
    #             all_edges.append([np2, np1])
    #     all_edges = np.array(all_edges)
    #     weights = [1] * len(all_edges)
    #     G = sparse.csr_matrix((weights, (all_edges[:, 0], all_edges[:, 1])), shape=(max_np + 1, max_np + 1))
    #     activate_score_page = pagerank(G, p=0.85)
    #     activate_score_page = {str(idx) : activate_score_page[idx] for idx in range(len(activate_score_page))}
    #     pickle.dump(activate_score_page, open('data/myexp/activate_score_page_gjq.pkl', 'wb'))
    ##############
    #############
    with open('data/myexp/active_links_gjq.txt', "r", encoding='utf-8') as rf:
        active_head_np_names = set([line.strip('\n') for line in rf.readlines()])
    np_act_score = defaultdict(list)
    for label,v in clusterentropy.items():
        if math.log(2-math.exp(-v),math.e)<math.exp(-1/(ite+1)) :
            clusterconfidence = 1 - (math.log(2 - math.exp(-v), math.e) / math.exp(-1 / (ite + 1)))
            for np in myclust2ent[label]:
                if np not in resdict.keys():
                    dst.writelines(np + '#####' + maxcount[label] + '#####' +
                                   str(clusterconfidence) + '\n')
                else:
                    if maxcount[label] in resdict[np]:
                        dst.writelines(np + '#####' + maxcount[label] + '#####' +
                                       str(clusterconfidence) + '\n')
                    else:
                        error += 1

        ##### 4 #####
        _true_false, _false = defaultdict(float), defaultdict(float)
        for np1, np2 in combinations(myclust2ent[label], 2):
            _true_false[np1] += confidence[np1] * confidence[np2]
            _true_false[np2] += confidence[np1] * confidence[np2]
            if result[np1] != result[np2]:
                _false[np1] += confidence[np1] * confidence[np2]
                _false[np2] += confidence[np1] * confidence[np2]
        #############

        for np in myclust2ent[label]:
            if np in active_head_np_names: continue
            ##### 1 #####
            np_act_score[np].append(confidence[np]) # v / confidence[np]
            #############
            ##### 2 #####
            np_act_score[np].append(
                1 - len(active_head_np_names & myclust2ent[label]) / len(myclust2ent[label])
            )
            #############
            ##### 3 #####
            np_act_score[np].append(activate_score_degree[np])    # activate_score3[np] / activate_score_degree[np] / activate_score_page[np]
            #############
            ##### 4 #####
            np_act_score[np].append(_false[np] / (_true_false[np]+1e-9))
            #############

    for idx in range(4):
        gjqgjq = [scores[idx] for scores in np_act_score.values()]
        for np, scores in np_act_score.items():
            np_act_score[np][idx] = (scores[idx] - min(gjqgjq)) / (max(gjqgjq) - min(gjqgjq))

    # scores[0] + scores[1] + scores[2] + scores[3]
    sorted_nps = sorted([(np, scores[0])
                         for np, scores in np_act_score.items()], key=lambda x:x[-1], reverse=True)
    with open('data/myexp/active_links_gjq.txt', "a", encoding='utf-8') as wf:
        wf.write("\n"+"\n".join([np for np, _ in sorted_nps[:(int(15530 * 0.01) + 1)]]))

#############################################################################################



def cluster_test(params, ite, isSub, triples, ent2id, id2ent, cluster_predict_list, true_ent2clust, true_clust2ent, print_or_not=False):
    sub_cluster_predict_list = []
    clust2ent = {}

    for eid in isSub.keys():
        sub_cluster_predict_list.append(cluster_predict_list[eid])

    for sub_id, cluster_id in enumerate(sub_cluster_predict_list):
        if cluster_id in clust2ent.keys():
            clust2ent[cluster_id].append(sub_id)
        else:
            clust2ent[cluster_id] = [sub_id]
    cesi_clust2ent = {}
    for rep, cluster in clust2ent.items():

        cesi_clust2ent[rep] = set(cluster)
    cesi_ent2clust = invertDic(cesi_clust2ent, 'm2os')

    clust2ent = {}
    for sub_id, cluster_id in enumerate(sub_cluster_predict_list):
        if cluster_id in clust2ent.keys():
            clust2ent[cluster_id].append(id2ent[sub_id])
        else:
            clust2ent[cluster_id] = [id2ent[sub_id]]
    myclust2ent = {}
    for rep, cluster in clust2ent.items():

        myclust2ent[rep] = set(cluster)

    generatelinkseed(myclust2ent, ent2id, id2ent,ite)#generate soft link seed for next iteration


    cesi_ent2clust_u = {}
    if params.use_assume:
        for trp in triples:
            sub_u, sub = trp['triple_unique'][0], trp['triple'][0]
            cesi_ent2clust_u[sub_u] = cesi_ent2clust[ent2id[sub]]
    else:
        for trp in triples:
            sub_u, sub = trp['triple_unique'][0], trp['triple_unique'][0]
            cesi_ent2clust_u[sub_u] = cesi_ent2clust[ent2id[sub]]

    cesi_clust2ent_u = invertDic(cesi_ent2clust_u, 'm2os')
    eval_results = evaluate(cesi_ent2clust_u, cesi_clust2ent_u, true_ent2clust, true_clust2ent)
    macro_prec, micro_prec, pair_prec = eval_results['macro_prec'], eval_results['micro_prec'], eval_results['pair_prec']
    macro_recall, micro_recall, pair_recall = eval_results['macro_recall'], eval_results['micro_recall'], eval_results['pair_recall']
    macro_f1, micro_f1, pair_f1 = eval_results['macro_f1'], eval_results['micro_f1'], eval_results['pair_f1']
    ave_prec = (macro_prec + micro_prec + pair_prec) / 3
    ave_recall = (macro_recall + micro_recall + pair_recall) / 3
    ave_f1 = (macro_f1 + micro_f1 + pair_f1) / 3
    model_clusters = len(cesi_clust2ent_u)
    model_Singletons = len([1 for _, clust in cesi_clust2ent_u.items() if len(clust) == 1])
    gold_clusters = len(true_clust2ent)
    gold_Singletons = len([1 for _, clust in true_clust2ent.items() if len(clust) == 1])
    if print_or_not:
        print('Ave-prec=', ave_prec, 'macro_prec=', macro_prec, 'micro_prec=', micro_prec,
              'pair_prec=', pair_prec)
        print('Ave-recall=', ave_recall, 'macro_recall=', macro_recall, 'micro_recall=', micro_recall,
              'pair_recall=', pair_recall)
        print('Ave-F1=', ave_f1, 'macro_f1=', macro_f1, 'micro_f1=', micro_f1, 'pair_f1=', pair_f1)
        print('Model: #Clusters: %d, #Singletons %d' % (model_clusters, model_Singletons))
        print('Gold: #Clusters: %d, #Singletons %d' % (gold_clusters, gold_Singletons))
        print()

    return ave_prec, ave_recall, ave_f1, macro_prec, micro_prec, pair_prec, macro_recall, micro_recall, pair_recall, \
           macro_f1, micro_f1, pair_f1, model_clusters, model_Singletons, gold_clusters, gold_Singletons
