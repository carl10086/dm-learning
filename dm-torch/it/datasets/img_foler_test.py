from datasets import load_dataset

train_set = load_dataset("imagefolder", data_dir="/root/autodl-tmp/dataset/imgs", drop_labels=False)
