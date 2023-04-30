from datasets import load_dataset

ds = load_dataset("imagefolder", data_dir="/root/autodl-tmp/dataset/imgs", drop_labels=False)
