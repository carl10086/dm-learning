import os
import json

from datasets import load_dataset


# ds = load_dataset("imagefolder", data_dir="/root/autodl-tmp/dataset/imgs", drop_labels=False)

def write_metadata():
    img_filenames = os.listdir("/root/autodl-tmp/dataset/imgs/train")
    img_metas = [json.dumps({
        "file_name": filename,
        "categories": ["库罗米", "动漫"]
    }) for filename in img_filenames]
    with open("/root/autodl-tmp/dataset/imgs/train/metadata.jsonl", "w") as f:
        f.write("\n".join(img_metas))


if __name__ == '__main__':
    # write_metadata()
    ds = load_dataset(
        "imagefolder",
        data_dir="/root/autodl-tmp/dataset/imgs",
        drop_labels=True,
    )

    labels = ["库罗米", "动漫"]
