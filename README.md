# Set-Aligning-Event-Temporal-Graph-Generation
This is the repository of the experimental code and data of "Set-Aligning Framework for Auto-Regressive Event Temporal Graph Generation" (NAACL 2024)

If our paper and code help, please consider adding the following reference in your research:
```
@misc{tan2024setaligning,
      title={Set-Aligning Framework for Auto-Regressive Event Temporal Graph Generation}, 
      author={Xingwei Tan and Yuxiang Zhou and Gabriele Pergola and Yulan He},
      year={2024},
      eprint={2404.01532},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

To repoduce the SAF results, follow the steps below one-by-one.


## Prepare the dataset
Download the NYT corpus and save it as *NYT_annotated* under this directory

## Process the raw documents with CAEVO
Download CAEVO source code from [https://github.com/nchambers/caevo](https://github.com/nchambers/caevo)
```
mkdir data/caevo_inputs

python get_nyt_data.py --select_from_ids_file data/train_file_ids.json --output_path data/caevo_inputs

python get_nyt_data.py --select_from_ids_file data/test_file_ids.json --output_path data/caevo_inputs

mkdir data/caevo_outputs

python run_caevo_on_dir.py --input-dir data/caevo_inputs --out-dir data/caevo_outputs
```

## Construct the target graphs
```
python get_target_graphs.py --input-dir data/caevo_outputs --select-file-path data/train_file_ids.json --output-path data/NYT_des_train.json --num-permu 4

python get_target_graphs.py --input-dir data/caevo_outputs --select-file-path data/test_file_ids.json --output-path data/NYT_des_test.json
```

## Prepare offset mapping for SPR caliberation
```
python prepare_offset.py --data_path data/NYT_des_train.json
```

## Training a flan-T5-base with set aligning framework
```
sh training_script.sh
```


