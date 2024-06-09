from transformers import AutoTokenizer, T5ForConditionalGeneration
import re
import json
from nltk.translate.bleu_score import corpus_bleu
from argparse import ArgumentParser
from datasets import Dataset
from torch.utils.data import DataLoader
import torch
import numpy as np
from tqdm import tqdm
from accelerate import Accelerator
import os


LOCAL_RANK = os.environ.get('LOCAL_RANK',-1)
# maximum generation length
MAX_LENGTH = 2048               


def parse_DOT(dot_str):
    # output: graph edge list
    if len(dot_str) < 12:
        return [], 0

    edges = dot_str[14:].split(';')
    key_set = set()
    graph = [] # graph edge list
    duplicate = 0
    for edge_str in edges:
        edge_str = re.sub('is_included', 'isincluded', edge_str)
        rel_list = re.findall(r'rel=([a-zA-Z]+)', edge_str)

        if len(rel_list) < 1:
            break

        rel = rel_list[0].lower()

        if rel not in ['after', 'before', 'includes', 'simultaneous', 'isincluded']: #['after', 'before']:
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
            duplicate += 1
        else:
            graph.append((event_1, rel, event_2))
            key_set.add(key)
            # print(event_1, rel, event_2)
    #print(f"Num of duplication: {duplicate}")
    return graph, duplicate


def preprocess_function(examples):
    inputs = [doc for doc in examples["document"]]
    model_inputs = tokenizer(inputs, max_length=MAX_LENGTH, truncation=True, padding=True)

    return model_inputs


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model-checkpoint", type=str, default="calib_model_dir")
    parser.add_argument("--test-path", type=str, default="data/NYT_des_test.json")
    parser.add_argument("--output-dir", type=str, default="out")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--generation", type=str, default="greedy")
    args = parser.parse_args()

    accelerator = Accelerator()
    accelerator.wait_for_everyone()

    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)
    model = T5ForConditionalGeneration.from_pretrained(args.model_checkpoint)

    graph_string_metrics = {}
    
    with open(args.test_path, 'r') as f:
        nyt_dev = json.loads(f.read())
    
    print("Validation documents:")
    print(len(nyt_dev))
    dev_input_dict = {
        'document': [nyt_dev[d]['document'] for d in nyt_dev],
        'summary': [nyt_dev[d]['target'] for d in nyt_dev],
        'doc_id': [d for d in nyt_dev],
        'id': [i for i in range(len(nyt_dev.keys()))]
    }
    dev_dataset = Dataset.from_dict(dev_input_dict)
    with accelerator.main_process_first():
        tokenized_dev_datasets = dev_dataset.map(preprocess_function, batched=True)
        print(f"{len(dev_dataset)}/{args.batch}, estimate {len(dev_dataset)/args.batch} iterations")

    eval_dataloader = DataLoader(tokenized_dev_datasets, batch_size=args.batch, shuffle=False)

    model, eval_dataloader, _, _ = accelerator.prepare(model, eval_dataloader, None, None)

    model.eval()
    total_generated = 0
    total_gold = 0
    test_out = {}
    predictions = []
    references = []
    total_duplicate = 0
    total_gold_node = 0
    for j, batch in tqdm(enumerate(eval_dataloader)):
        input_ids = torch.cat(batch['input_ids']).view(-1, args.batch).T
        att_mask = torch.cat(batch['attention_mask']).view(-1, args.batch).T

        with torch.no_grad():
            if args.generation == "greedy":
                outputs = accelerator.unwrap_model(model).generate(
                                    input_ids=input_ids.cuda(),
                                    attention_mask=att_mask.cuda(), 
                                    max_length=MAX_LENGTH,
                                    num_beams=1,
                                    early_stopping=True
                                )
            elif args.generation == "beam":
                outputs = accelerator.unwrap_model(model).generate(
                                        input_ids=input_ids.cuda(),
                                        attention_mask=att_mask.cuda(), 
                                        max_length=MAX_LENGTH,
                                        num_beams=5,
                                        early_stopping=True
                                    )
            elif args.generation == "nucleus":
                outputs = accelerator.unwrap_model(model).generate(
                                        input_ids=input_ids.cuda(),
                                        attention_mask=att_mask.cuda(),
                                        max_length=MAX_LENGTH,
                                        do_sample=True,
                                        top_p=0.92,
                                        top_k=0
                                    )
                
        out_str = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        for i in range(len(batch['summary'])):
            gold_graph, _ = parse_DOT(batch['summary'][i])

            gold_node_set = set()
            for gold_e1, gold_rel, gold_e2 in gold_graph:
                gold_node_set.add(gold_e1)
                gold_node_set.add(gold_e2)
            total_gold_node += len(gold_node_set)

            total_gold += len(gold_graph)
            generated_graph, duplicate = parse_DOT(out_str[i])
            total_duplicate += duplicate
            total_generated += len(generated_graph)
            test_out[batch['doc_id'][i]] = {
                        "document": batch['document'],
                        "gold": gold_graph,
                        "generated": generated_graph,
                        "out_string": out_str[i]
                    }

        predictions += [text.lower() for text in out_str]
        references += [text.lower() for text in batch['summary']]

        
    print(f"Total generated: {total_generated}\nTotal gold: {total_gold}")
    with open(f'{args.output_dir}/{args.model_checkpoint}-nyt-test-{LOCAL_RANK}.json', 'w') as f: 
        # if multiple processes/gpus, each process will write to a different file
        f.write(json.dumps(test_out, indent=4))
    print(f"Total gold node: {total_gold_node}")
    
    if accelerator.is_main_process:
        # if multiple processes/gpus, each process will write to a different file
        print(f"Evaluation outputs saved at {args.output_dir}/{args.model_checkpoint}-nyt-test.json")
