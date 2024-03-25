import json
from os import listdir
from argparse import ArgumentParser
import xml.etree.ElementTree as ET
import re
#import spacy
#spacy.prefer_gpu()
#nlp = spacy.load("en_core_web_sm")


FILTER_VERBS = ["said", "say", "had", "made", "told", "appear", "be", "become", "do", "have", "seem", "get", "give", "go", "have", "keep", "make", "put", "set", "take", "argue", "claim", "suggest", "tell", "says"]
FILTER_VERBS = set(FILTER_VERBS)
PUNC_LIST = ['.', ',', "'", '"', '?', '!', ';', ':', '#', '$', '%', '&', '(', ')', '*', '+', '-', '/', '<', '=', '>', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~']
PUNC_SET = set(PUNC_LIST)


import json
import re 


def parse_DOT(dot_str):
    # output: graph edge list
    if len(dot_str) < 15:
        return [], 0

    edges = dot_str[15:].split(';')
    key_set = set()
    graph = [] # graph edge list
    for edge_str in edges:
        edge_str = re.sub('is_included', 'isincluded', edge_str)
        rel_list = re.findall(r'rel=([a-zA-Z]+)', edge_str)
        if len(rel_list) < 1:
            break
        rel = rel_list[0].lower()
        if rel not in ['after', 'before', 'includes', 'simultaneous', 'isincluded']:
            continue

        if rel == 'isincluded':
            rel = 'is_included'
        event_pair = edge_str.split('[rel=')[0]
        if len(event_pair.split('--')) < 2:
            continue
        event_1 = event_pair.split(' -- ')[0].lower()
        event_2 = event_pair.split(' -- ')[1].lower()
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

        key = f"{event_1}||{rel}||{event_2}"
        if key in key_set:
            continue
        else:
            graph.append((event_1, rel, event_2))
            key_set.add(key)

    return graph


def remove_reci(data_dict):
    new_data = []
    total_events = 0
    total_relations = 0
    after_total_relations = 0
    after_total_events = 0
    type_dict = {
            'includes': 0,
            'before': 0,
            'after': 0,
            'is_included': 0,
            'simultaneous': 0
        }

    total_degree = 0
    for doc_dict in data_dict:
        event_set = set()
        relset = set()

        original_target = data_dict[doc_dict]['target']
        graph = parse_DOT(original_target)

        after_event_set = set()
        degree_count = {}
        
        new_target = "strict graph {\n"
        for edge in graph:
            event_set.add(edge[0])
            event_set.add(edge[2])
            total_relations += 1

            if edge[1] == 'is_included':
                this_edge =  f"\"{edge[2]}\" -- \"{edge[0]}\" [rel=includes];\n"
            elif edge[1] == 'after':
                this_edge = f"\"{edge[2]}\" -- \"{edge[0]}\" [rel=before];\n"
            else:
                this_edge = f"\"{edge[0]}\" -- \"{edge[2]}\" [rel={edge[1]}];\n"
            
            if this_edge in relset:
                continue
            else:
                new_target += this_edge
                relset.add(this_edge)

                after_event_set.add(' '.join(edge[0].split()))
                after_event_set.add(' '.join(edge[2].split()))
                type_dict[edge[1]] += 1

                degree_count[edge[0]] = degree_count.get(edge[0], 0) + 1
                degree_count[edge[2]] = degree_count.get(edge[2], 0) + 1

        total_degree += sum(degree_count.values())
        total_events += len(event_set)
        after_total_relations += len(relset)
        after_total_events += len(after_event_set)

        new_target += "}"

        data_dict[doc_dict]['target'] = new_target

    return data_dict


def add_permutation(train_dict, upper_limit=5):
    # maximum permutation is the upper_limit
    new_train_dict = {}

    for doc_id in train_dict:
        count = 0
        origin_target = train_dict[doc_id]['target']

        edge_list = origin_target[14:-1].split('\n')

        for permu in permutations(edge_list):
            if count >= upper_limit:
                break

            new_train_dict[f"{doc_id}+{count}"] = {
                "document": train_dict[doc_id]['document'],
                "target": "strict graph {\n" + '\n'.join(permu) + "\n}"
            }
            count += 1

    return new_train_dict


def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text)
    return text


def parse_xml(path):
    with open(path, 'r') as f:
        tree = ET.ElementTree(ET.fromstring(f.read()))

    # get root element
    root = tree.getroot()

    entries = []
    tlinks = []
    # iterate news items
    for file in root.findall('{http://chambers.com/corpusinfo}file'):
        for entry in file.findall('{http://chambers.com/corpusinfo}entry'):
            sentence = entry.find('{http://chambers.com/corpusinfo}sentence').text
            event_list = []
            #for token in nlp(sentence):
            #    print(token.pos, spacy.symbols.VERB)
            tokens = entry.find('{http://chambers.com/corpusinfo}tokens')
            tk_list = []
            parse_tree = entry.find('{http://chambers.com/corpusinfo}parse').text
            dependency = entry.find('{http://chambers.com/corpusinfo}deps').text
            if type(dependency) != type('what'):
                dependency_list = []
            else:
                dependency_list = dependency.split("\n")
            for t in tokens.findall('{http://chambers.com/corpusinfo}t'):
                #tk = re.findall(r"\"([^\s]+)\"", t.text)
                count = 0
                reverse_count = 0
                for i in range(len(t.text)):
                    if t.text[i] == "\"":
                        count += 1
                    if count == 3:
                        start = i+1
                    
                    if t.text[len(t.text)-1-i] == "\"":
                        reverse_count += 1
                    if reverse_count == 3:
                        end = len(t.text)-1-i
                tk_list.append(t.text[end:start])
            for event in entry.findall('{http://chambers.com/corpusinfo}events/{http://chambers.com/corpusinfo}event'):
                event_list.append(event.attrib)
            entries.append(
                {
                    "sentence": sentence,
                    "tokens": tk_list,
                    "events": event_list,
                    "parse_tree": parse_tree,
                    "dependency": dependency_list
                }
            )
        for tlink in file.findall('{http://chambers.com/corpusinfo}tlink'):
            tlinks.append(tlink.attrib)
    return entries, tlinks


if __name__ == "__main__":
    parser = ArgumentParser(description='Turn caevo files to seq2seq document target pairs')
    parser.add_argument("--input-dir", type=str, default="NYT_xml",
            help="the path to the file directory")
    parser.add_argument("--select-file-path", type=str, default="")
    parser.add_argument("--output-path", type=str, default="NYT_des_train.json")
    parser.add_argument("--num-permu", type=int, default=0)
    args = parser.parse_args()

    file_list = listdir(args.input_dir)

    total_event_num = 0
    out_dict = {}
    relation_num = {
        "before": 0,
        "after": 0,
        "is_included": 0,
        "simultaneous": 0,
        "includes": 0
    }
    total_tk_len = 0

    if len(args.select_file_path) > 0:
        with open(args.select_file_path, 'r') as f:
            selected_ids = json.loads(f.read())

    for file_name in file_list:
        if file_name.split('.')[-1] == 'xml':
            if len(args.select_file_path) > 0 and file_name.split('.')[0] not in selected_ids:
                continue

            entries, tlinks = parse_xml(f"{args.input_dir}/{file_name}")
            document = ' '.join([s_dict["sentence"] for s_dict in entries])
            # clean document to reduce the token length
            # document = clean_text(document)
            total_tk_len += len(document.split())
            ei2str = {} 
            ei2prefixsubfix = {}
            ei2sid = {}

            for sid, s_dict in enumerate(entries): 
                #total_tk_len += len(s_dict["tokens"])
                for event_dict in s_dict["events"]:
                    if event_dict["string"] not in FILTER_VERBS: # excluding light verbs
                        ei2str[event_dict["eiid"]] = event_dict["string"]
                        ei2sid[event_dict["eiid"]] = sid
                        offset = int(event_dict["offset"])
                        this_dep_key = f"{event_dict['string']}-{offset}"
                        ps_dict = {
                            'nsubj': [],
                            'dobj': []
                        }
                        for dep_element in s_dict["dependency"]:
                            dep_type = dep_element.split('(')[0]
                            root_node = dep_element.split('(')[1].split(',')[0]
                            child_node = dep_element.split('(')[1].split(', ')[1][:-1]
                            if dep_type in ['nsubj', 'dobj'] and root_node==this_dep_key:
                                ps_dict[dep_type].append(child_node.split('-')[0])
                        eventprefixsuffix = ""
                        if len(ps_dict['nsubj']) > 0:
                            eventprefixsuffix += ' '.join(ps_dict['nsubj']) + " "
                        eventprefixsuffix += event_dict["string"]
                        if len(ps_dict['dobj']) > 0:
                            eventprefixsuffix += " " + ' '.join(ps_dict['dobj'])

                        ei2prefixsubfix[event_dict["eiid"]] = eventprefixsuffix

            total_event_num += len(ei2sid)
            target_string = "strict graph {"
            for t_dict in tlinks:
                if t_dict["type"] == "ee" and ei2str.get(t_dict["event1"]) is not None and ei2str.get(t_dict["event2"]) is not None:
                    #this_head = ei2str[t_dict["event1"]]
                    #this_tail = ei2str[t_dict["event2"]]
                    this_head = ei2prefixsubfix[t_dict["event1"]]
                    this_tail = ei2prefixsubfix[t_dict["event2"]]
                    this_rel = t_dict["relation"].lower()
                    if this_rel != "vague":
                        target_string += f"\n\"{this_head}\" -- \"{this_tail}\"  [rel={this_rel}];"
                        relation_num[this_rel] += 1

            if target_string == "strict graph {":
                continue

            target_string += '\n}'
    
            doc_id = file_name.split('.txt')[0]
            out_dict[doc_id] = {
                "document": document,
                "target": target_string
            }
    
    total_doc_num = len(out_dict)
    total_rel = 0
    for rel_type in relation_num:
        total_rel += relation_num[rel_type]

    print(f"{total_doc_num} documents in total\n{total_event_num} events in total\n{total_rel} relation links in total")
    print(relation_num)
    print(f"Average event {total_event_num/total_doc_num}\nAverage links {total_rel/total_doc_num}")
    print(f"Average number of tokens per document: {total_tk_len/total_doc_num}")

    out_dict = remove_reci(out_dict)

    if args.num_permu > 0:
        out_dict = add_permutation(out_dict, args.num_permu)

    with open(args.output_path, 'w') as f:
        f.write(json.dumps(out_dict, indent=4))
