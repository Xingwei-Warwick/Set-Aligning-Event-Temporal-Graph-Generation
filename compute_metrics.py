import re
import json
from query_openie import Allen_api, get_verbs
import networkx as nx
from argparse import ArgumentParser
import evaluate


if __name__ == "__main__":
    parser = ArgumentParser(description='Evaluate a graph')
    parser.add_argument("--path", type=str, default="out/calib_model_dir-nyt-test.json",
            help="the path to the processed evaluation output file")
    parser.add_argument("--num-split", type=int, default=1, help="number of splits if DDP was used in evaluation")
    args = parser.parse_args()

    if args.num_split == 1:
        with open(args.path, 'r') as f:
            test_dict = json.loads(f.read())
    elif args.num_split > 1:
        path_base = args.path.split(".json")[0]
        test_dict = {}
        for i in range(args.num_split):
            with open(f"{path_base}-{i}.json", 'r') as f:
                test_dict.update(json.loads(f.read()))
    else:
        exit(0)

    total_generated = 1e-5
    total_gold = 0
    tp = 1e-5
    fp = 0
    fn = 0
    total_generated_node = 1e-5
    total_gold_node = 0
    node_tp = 1e-5
    node_fp = 0
    node_fn = 0
    gold_rel_dist = {}
    gen_rel_dist = {}
    total_gen_degree_count = 0
    
    for doc_id in test_dict:
        generated_node_set = set()
        gold_node_set = set()
        gold_graph = test_dict[doc_id]['gold']
        total_gold += len(gold_graph)
        edge_list = test_dict[doc_id]['generated']
        total_generated += len(edge_list)
        degree_count = {}

        retrieved_gold = set()
        for gold_e1, gold_rel, gold_e2 in gold_graph:
            if gold_e1[-1] == ' ' or gold_e1[-1] == '\n':
                gold_e1 = gold_e1[:-1]
            if gold_e2[-1] == ' ' or gold_e2[-1] == '\n':
                gold_e2 = gold_e2[:-1]
            gold_node_set.add(gold_e1)
            gold_node_set.add(gold_e2)
            gold_rel_dist[gold_rel] = gold_rel_dist.get(gold_rel, 0) + 1
        for e1, rel, e2 in edge_list:
            generated_node_set.add(e1)
            generated_node_set.add(e2)
            found = False

            gen_rel_dist[rel] = gen_rel_dist.get(rel, 0) + 1

            gen_e1_words = " ".join(e1.split())
            gen_e2_words = " ".join(e2.split())
            degree_count[gen_e1_words] = degree_count.get(gen_e1_words, 0) + 1
            degree_count[gen_e2_words] = degree_count.get(gen_e2_words, 0) + 1
            e1_word_set = set(e1.split())
            e2_word_set = set(e2.split())
            for gold_e1, gold_rel, gold_e2 in gold_graph:
                gold_e1_words = " ".join(gold_e1.split())
                gold_e2_words = " ".join(gold_e2.split())
                gold_e1_word_set = set(gold_e1.split())
                gold_e2_word_set = set(gold_e2.split())

                if rel == gold_rel and len(gold_e1_word_set-e1_word_set)==0 and len(gold_e2_word_set-e2_word_set)==0:
                    key = f"{gold_e1}||{gold_rel}||{gold_e2}"
                    if key not in retrieved_gold:
                        retrieved_gold.add(key)
                        found = True
                        break
                
                if gold_rel == "simultaneous" and rel == "simultaneous" and len(gold_e1_word_set-e2_word_set)==0 and len(gold_e2_word_set-e1_word_set)==0:
                    key = f"{gold_e1}||{gold_rel}||{gold_e2}"
                    if key not in retrieved_gold:
                        retrieved_gold.add(key)
                        found = True
                        break

            if found:
                tp += 1
            else:
                fp += 1
        fn += len(gold_graph) - len(retrieved_gold)

        ######### compute node metrics
        total_generated_node += len(generated_node_set)
        total_gold_node += len(gold_node_set)
        retrieved_gold_node = set()

        for generated_node in generated_node_set:
            found = False
            node_words = " ".join(generated_node.split())
            node_word_set = set(generated_node.split())
            for gold_node in gold_node_set:
                gold_node_words = " ".join(gold_node.split())
                gold_node_word_set = set(gold_node.split())
                if len(gold_node_word_set-node_word_set)==0 and gold_node not in retrieved_gold_node:
                    found = True
                    retrieved_gold_node.add(gold_node)
            if found:
                node_tp += 1
            else:
                node_fp += 1
        node_fn += len(gold_node_set) - len(retrieved_gold_node)
        total_gen_degree_count += sum(degree_count.values())


    print("="*10 + "Edge Metrics" + "="*10)
    print(f"Total generated: {total_generated}\nTotal gold: {total_gold}\nMatched: {tp}")
    recall = tp / total_gold
    precision = tp / total_generated
    f1 = 2 * recall * precision / (recall + precision)
    print(f"F1: {f1}, Recall: {recall}, Precision: {precision}")
        
    print("="*10 + "Node Metrics" + "="*10)
    print(f"Total generated: {total_generated_node}\nTotal gold: {total_gold_node}\nMatched: {node_tp}")
    recall = node_tp / total_gold_node
    precision = node_tp / total_generated_node
    f1 = 2 * recall * precision / (recall + precision)
    print(f"F1: {f1}, Recall: {recall}, Precision: {precision}")
    print("gold graph temporal rel type distribution:")
    print(gold_rel_dist)
    print("generated graph temporal rel type distribution:")
    print(gen_rel_dist)

    print("average degree count:")
    print(total_gen_degree_count / total_generated_node)