import datetime
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
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)

label_col_name = 'labels'

# use proxy
os.environ['HTTP_PROXY'] = 'http://192.168.126.12:12798'
os.environ['HTTPS_PROXY'] = 'http://192.168.126.12:12798'

metric = load_metric("accuracy")
model_name_or_path = 'google/vit-base-patch16-224-in21k'
feature_extractor = ViTFeatureExtractor.from_pretrained(model_name_or_path)

# 根据 feature extractor 的 size，重新定义 train 和 valid train 的 transforms
normalize = Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
size = feature_extractor.size['height']
train_transforms = Compose(
    [
        RandomResizedCrop(size),
        RandomHorizontalFlip(),
        ToTensor(),
        normalize,
    ]
)

val_transforms = Compose(
    [
        Resize(size),
        CenterCrop(size),
        ToTensor(),
        normalize,
    ]
)


def preprocess_train(example_batch):
    """Apply train_transforms across a batch."""
    example_batch['pixel_values'] = [
        train_transforms(image.convert("RGB")) for image in example_batch["image"]
    ]
    return example_batch


def preprocess_val(example_batch):
    """Apply val_transforms across a batch."""
    example_batch["pixel_values"] = [val_transforms(image.convert("RGB")) for image in example_batch["image"]]
    return example_batch


def compute_metrics(eval_pred):
    """
    Next, we need to define a function for how to compute the metrics from the predictions,
    which will just use the metric we loaded earlier.
    The only preprocessing we have to do is to take the argmax of our predicted logits:
    """
    return metric.compute(
        predictions=np.argmax(eval_pred.predictions, axis=1), references=eval_pred.label_ids)


def collate_fn(batch):
    """
    We also define a collate_fn, which will be used to batch examples together. Each batch consists of 2 keys, namely pixel_values and labels.
    """
    return {
        'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
        'labels': torch.tensor([x[label_col_name] for x in batch])
    }


if __name__ == '__main__':
    ds = load_dataset("imagefolder", data_dir="/root/autodl-tmp/dataset/star_v1/dataset", drop_labels=False)

    # 不同的模型随着 hugging face/transformers 的更新，可能会有不同的 label_col_name
    if label_col_name != 'label':
        ds = ds.rename_column('label', label_col_name)

    # 取出自定义的 labels 映射
    labels = ds['train'].features[label_col_name].names

    # ignore_mismatched_sizes = True,  # provide this in case you're planning to fine-tune an already fine-tuned checkpoint
    model = ViTForImageClassification.from_pretrained(
        model_name_or_path,
        num_labels=len(labels),
        id2label={str(i): c for i, c in enumerate(labels)},
        label2id={c: str(i) for i, c in enumerate(labels)}
    )

    train_ds = ds['train']
    val_ds = ds['validation']

    # with_transform
    train_ds.set_transform(preprocess_train)
    val_ds.set_transform(preprocess_val)

    version = datetime.datetime.now().strftime("%m%d_%H%M")
    output_dir = f"/root/autodl-tmp/output/vit-base-patch16-224-starv2-{version}"

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=32,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=100,
        fp16=True,
        logging_steps=10,
        learning_rate=2e-4,
        save_total_limit=2,
        remove_unused_columns=False,
        push_to_hub=False,
        report_to=['tensorboard'],
        load_best_model_at_end=True,
        # metric_for_best_model="accuracy",
    )

    trainer = Trainer(
        model,
        training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        # This is only to make sure the feature extractor configuration file (stored as JSON) will also be uploaded to the repo on the hub.
        tokenizer=feature_extractor,
        compute_metrics=compute_metrics,
        data_collator=collate_fn,
    )

    train_results = trainer.train()
    # rest is optional but nice to have
    trainer.save_model()
    trainer.log_metrics("train", train_results.metrics)
    trainer.save_metrics("train", train_results.metrics)
    trainer.save_state()

    # We can check with the evaluate method that our Trainer did reload the best model properly (if it was not the last one):
    metrics = trainer.evaluate(val_ds)
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)
