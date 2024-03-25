from caliberation_loss import SequenceParser
from argparse import ArgumentParser
from transformers import AutoTokenizer
import json


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path to the training data.")
    parser.add_argument("--model_name_or_path", type=str, default="google/flan-t5-base", help="Model name or path.")
    args = parser.parse_args()

    parser = SequenceParser(AutoTokenizer.from_pretrained(args.model_name_or_oath), max_length=2048)
    with open(args.data_path, 'r') as f:
        train_data = json.loads(f.read())

    out_path = args.data_path.replace(".json", "_offset.json")
    
    offset_dict = {}
    for doc_id in train_data:
        target = train_data[doc_id]['target']
        offset_dict[doc_id], _ = parser.parse(target)
    
    with open(out_path, 'w') as f:
        f.write(json.dumps(offset_dict, indent=4))