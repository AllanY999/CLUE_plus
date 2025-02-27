import pickle

# np2id = pickle.load(open('self.np2id', 'rb'))
# id2np = pickle.load(open('self.id2np', 'rb'))
# isSub = pickle.load(open('self.isSub', 'rb'))
# newid2ent = pickle.load(open('nplist', 'rb'))
# relation_view_embed = []
# count=0
# for ent in newid2ent:
#     id = np2id[ent]
#     if id in isSub:
#         count+=1
#
# print(id2np)

with open('data/myexp/head_np_links_gjq.txt', "r", encoding='utf-8') as rf:
    head_np_links_dict = {line.strip('\n').split('#####')[0]: line.strip('\n').split('#####')[1]
                          for line in rf.readlines()}

with open('data/myexp/active_links_gjq.txt', "r", encoding='utf-8') as rf:
    active_head_np_names = [line.strip('\n') for line in rf.readlines()]
count=0
for i in active_head_np_names:
    if i not in head_np_links_dict.keys():
        print(i)
        count+=1
print(count)