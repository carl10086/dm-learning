import os
import shutil
import numpy as np
from datasets import load_metric
from datasets import load_dataset
import torch
from transformers import TrainingArguments
from transformers import ViTFeatureExtractor
from transformers import ViTForImageClassification
from transformers import Trainer


def read_lines(file):
    with open(file, "r") as f:
        lines = f.readlines()
        lines = [x.strip('\n') for x in lines]
        return np.array(lines)


# use proxy
os.environ['HTTP_PROXY'] = 'http://192.168.126.12:12798'
os.environ['HTTPS_PROXY'] = 'http://192.168.126.12:12798'

DIR = "/root/autodl-tmp/dataset/star_v1"

model_name_or_path = 'google/vit-base-patch16-224-in21k'
feature_extractor = ViTFeatureExtractor.from_pretrained(model_name_or_path)


def mkdir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def cp(from_path, to_path):
    """
    copy file from from_path -> to_path
    """
    shutil.copy(from_path, to_path)


def cp_dataset():
    for line in read_lines(f'{DIR}/train.txt'):
        print(line)
        label = line.split("_")[0]
        img_path = line.split("/")[-1]
        dir_path = f'{DIR}/dataset/train/{label}'
        mkdir(dir_path)
        cp(
            from_path=f'{DIR}/pics/{img_path}',
            to_path=f'{dir_path}/{img_path}'
        )


def transform(example_batch):
    # Take a list of PIL images and turn them to pixel values
    inputs = feature_extractor([x.convert("RGB") for x in example_batch['image']], return_tensors='pt')

    # Don't forget to include the labels!
    inputs['labels'] = example_batch['labels']
    return inputs


def collate_fn(batch):
    return {
        'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
        'labels': torch.tensor([x['labels'] for x in batch])
    }


metric = load_metric("accuracy")


def compute_metrics(p):
    return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)


if __name__ == '__main__':
    # cp_dataset()
    ds = load_dataset("imagefolder", data_dir="/root/autodl-tmp/dataset/star_v1/dataset", drop_labels=False)
    ds = ds.rename_column('label', 'labels')

    prepared_ds = ds.with_transform(transform)

    # prepared_ds['train'][0:2]['pixel_values'].shape
    # should be (2,3, 224, 224)

    labels = ds['train'].features['labels'].names

    model = ViTForImageClassification.from_pretrained(
        model_name_or_path,
        num_labels=len(labels),
        id2label={str(i): c for i, c in enumerate(labels)},
        label2id={c: str(i) for i, c in enumerate(labels)}
    )

    training_args = TrainingArguments(
        output_dir="./vit-base-starv1",
        per_device_train_batch_size=16,
        evaluation_strategy="steps",
        num_train_epochs=4,
        fp16=True,
        save_steps=100,
        eval_steps=100,
        logging_steps=10,
        learning_rate=2e-4,
        save_total_limit=2,
        remove_unused_columns=False,
        push_to_hub=False,
        report_to=['tensorboard'],
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        train_dataset=prepared_ds["train"],
        eval_dataset=prepared_ds["validation"],
        tokenizer=feature_extractor,
    )
    train_results = trainer.train()
    trainer.save_model()
    trainer.log_metrics("train", train_results.metrics)
    trainer.save_metrics("train", train_results.metrics)
    trainer.save_state()

    metrics = trainer.evaluate(prepared_ds['validation'])
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)
