import re
from transformers import AutoTokenizer
import json
import numpy as np
from scipy.spatial import distance


class SequenceParser(object):
    def __init__(self, tokenizer, max_length=1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def parse(self, dot_str):
        # can tokenize all the sequences first
        _, duplicate, offset_list = parse_DOT(dot_str)


        tokens = self.tokenizer(dot_str, padding=True, truncation=True, max_length=self.max_length, return_offsets_mapping=True)
        offset_mapping = tokens['offset_mapping']
        

        tk_offset_list = []
        for event_1_offset, rel_offset, event_2_offset in offset_list:
            event_1_tk_offset = []
            event_2_tk_offset = []
            rel_tk_offset = []
            for i in range(len(offset_mapping)):
                if offset_mapping[i][0] >= event_1_offset[0] and offset_mapping[i][1] <= event_1_offset[1]:
                    event_1_tk_offset.append(i)
                elif offset_mapping[i][0] >= event_2_offset[0] and offset_mapping[i][1] <= event_2_offset[1]:
                    event_2_tk_offset.append(i)
                elif offset_mapping[i][0] >= rel_offset[0] and offset_mapping[i][1] <= rel_offset[1]:
                    rel_tk_offset.append(i)
            if len(event_1_tk_offset)==0 or len(event_2_tk_offset)==0 or len(rel_tk_offset)==0:
                break
            tk_offset_list.append((event_1_tk_offset, rel_tk_offset, event_2_tk_offset))
        return (tk_offset_list, duplicate)
    

def parse_DOT(dot_str):
    dot_str = dot_str.lower()
    # output: graph edge list
    if len(dot_str) < 12:
        return [], 0, []

    edges = dot_str[12:].split(';')
    key_set = set()
    graph = [] # graph edge list
    duplicate = 0
    offset_list = []
    for edge_str in edges:
        
        edge_str_offset_start = dot_str.find(edge_str)

        edge_str = re.sub('is_included', 'issincluded', edge_str)
        rel_list = re.findall(r'rel=([a-zA-Z]+)', edge_str)
        if len(rel_list) < 1:
            break
        rel = rel_list[0]
        if rel not in ['after', 'before', 'includes', 'simultaneous', 'issincluded']: #['after', 'before']:
            continue
        
        rel = rel.strip()
        event_pair = edge_str.split('[rel=')[0]
        if len(event_pair.split(' -- ')) < 2:
            continue

        rel_offset_start = edge_str.find(f"[rel={rel}") + len("[rel=")
        rel_offset = (edge_str_offset_start + rel_offset_start, edge_str_offset_start + rel_offset_start + len(rel))
        if rel == 'issincluded':
            rel = 'is_included'
        event_1 = event_pair.split(' -- ')[0].lower()
        event_2 = event_pair.split(' -- ')[1].lower()
        if len(event_1)==0 or len(event_2)==0:
            continue
        if event_1[0] == ' ':
            event_1 = event_1[1:]
        event_1 = re.sub(r'\"', '', event_1)
        event_2 = re.sub(r'\"', '', event_2)
        event_1 = re.sub(r'\n', '', event_1)
        event_2 = re.sub(r'\n', '', event_2)
        if len(event_1)==0 or len(event_2)==0:
            continue
        if event_1==" " or event_2==" ":
            continue
        if event_1[0] == ' ':
            event_1 = event_1[1:]
        if event_2[0] == ' ':
            event_2 = event_2[1:]
        if event_1[-1] == ' ':
            event_1 = event_1[:-1]
        if event_2[-1] == ' ':
            event_2 = event_2[:-1]
        event_1 = event_1.strip()
        event_2 = event_2.strip()

        event_1 = '\n'+ event_1
        event_2 = ' ' + event_2
        event_1_offset_start = edge_str.find(event_1)
        event_1_offset = (edge_str_offset_start + event_1_offset_start, edge_str_offset_start + event_1_offset_start + len(event_1))
        event_2_offset_start = edge_str[event_1_offset_start+len(event_1):].find(event_2) + event_1_offset_start + len(event_1)
        event_2_offset = (edge_str_offset_start + event_2_offset_start, edge_str_offset_start + event_2_offset_start + len(event_2))

        key = f"{event_1}||{rel}||{event_2}"
        if key in key_set:
            duplicate += 1
        else:
            graph.append((event_1, rel, event_2))
            key_set.add(key)
            offset_list.append((event_1_offset, rel_offset, event_2_offset))

    return graph, duplicate, offset_list


def get_set_embs(offset_list, embs):
    set_embs = []
    for offsets in offset_list:
        if len(offsets) != 3:
            continue
        event_1_tk_offset, rel_tk_offset, event_2_tk_offset = offsets
        if max(event_1_tk_offset+rel_tk_offset+event_2_tk_offset) >= len(embs):
            continue
        event_1 = embs[np.array(event_1_tk_offset)].mean(0)
        rel = embs[np.array(rel_tk_offset)].mean(0)
        event_2 = embs[np.array(event_2_tk_offset)].mean(0)
        edge_emb = np.concatenate([event_1, rel, event_2], axis=-1)
        set_embs.append(edge_emb)

    if len(set_embs) == 0:
        return np.array([])
    else:
        return np.stack(set_embs)


def hausdorff_distance(set1, set2):
    if len(set1)==0 or len(set2)==0:
        print("Empty set!")
        return 0.
    
    distance_matrix = np.zeros((len(set1), len(set2)))
    for i in range(len(set1)):
        for j in range(len(set2)):
            distance_matrix[i][j] = distance.cosine(set1[i], set2[j])
    
    hausdorff = np.min(distance_matrix, 1).mean() + np.min(distance_matrix, 0).mean()
    return hausdorff


def set_size_loss(gold_set, gen_set):
    return abs(len(gold_set)-len(gen_set))/len(gold_set)
