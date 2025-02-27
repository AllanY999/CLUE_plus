import pickle

with open('data/myexp/head_np_links_gjq.txt', "r", encoding='utf-8') as rf:
    head_np_links_dict = {line.strip('\n').split('#####')[0]: line.strip('\n').split('#####')[1]
                          for line in rf.readlines()}

with open('data/myexp/active_links_gjq.txt', "r", encoding='utf-8') as rf:
    active_head_np_names = [line.strip('\n') for line in rf.readlines()]

with open('trainlinks.txt', "w", encoding='utf-8') as wf:
    for np_name in active_head_np_names:
        wf.write(np_name+'#####'+head_np_links_dict[np_name]+'\n')