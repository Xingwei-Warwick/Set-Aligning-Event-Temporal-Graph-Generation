import json


with open('data/NYT_des_human_temp.json', 'r') as f:
    test_dict = json.loads(f.read())

with open('data/human_gold_graphs.json', 'r') as f:
    human_gold_graphs = json.loads(f.read())

for doc_id in test_dict:
    test_dict[doc_id]['target'] = human_gold_graphs[doc_id]['target']

with open('data/NYT_des_human.json', 'w') as f:
    f.write(json.dumps(test_dict, indent=4))