# Set-Aligning-Event-Temporal-Graph-Generation
This is the repository of the experimental code and data of "Set-Aligning Framework for Auto-Regressive Event Temporal Graph Generation" (NAACL 2024)

If our paper and code help, please consider adding the following reference in your research:
```
@inproceedings{tan-etal-2024-set,
    title = "Set-Aligning Framework for Auto-Regressive Event Temporal Graph Generation",
    author = "Tan, Xingwei  and
      Zhou, Yuxiang  and
      Pergola, Gabriele  and
      He, Yulan",
    editor = "Duh, Kevin  and
      Gomez, Helena  and
      Bethard, Steven",
    booktitle = "Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)",
    month = jun,
    year = "2024",
    address = "Mexico City, Mexico",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.naacl-long.214",
    pages = "3872--3892",
}
```

To repoduce the SAF results, follow the steps below one-by-one.


## Prepare the dataset
Download the NYT corpus and save it as *NYT_annotated* under this directory

## Process the raw documents with CAEVO
Download CAEVO source code from [https://github.com/nchambers/caevo](https://github.com/nchambers/caevo)
```
mkdir data/caevo_inputs

python get_nyt_data.py --select_from_ids_file data/train_file_ids.json --output_dir data/caevo_inputs

python get_nyt_data.py --select_from_ids_file data/test_file_ids.json --output_dir data/caevo_inputs

python get_nyt_data.py --select_from_ids_file data/nyt_human_ids.json --output_dir data/caevo_inputs

mkdir data/caevo_outputs

python run_caevo_on_dir.py --input-dir data/caevo_inputs --out-dir data/caevo_outputs
```

## Construct the target graphs
```
python get_target_graphs.py --input-dir data/caevo_outputs --select-file-path data/train_file_ids.json --output-path data/NYT_des_train.json --num-permu 4

python get_target_graphs.py --input-dir data/caevo_outputs --select-file-path data/test_file_ids.json --output-path data/NYT_des_test.json

python get_target_graphs.py --input-dir data/caevo_outputs --select-file-path data/nyt_human_ids.json --output-path data/NYT_des_human_temp.json

python get_human_test.py
```

## Prepare offset mapping for SPR caliberation
```
python prepare_offset.py --data_path data/NYT_des_train.json
```

## Training a flan-T5-base with set aligning framework
```
sh training_script.sh
```

## Run the evaluation script
```
sh eval_script.sh
```
