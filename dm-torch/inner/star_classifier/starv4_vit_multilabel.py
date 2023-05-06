import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from transformers import EvalPrediction
import torch
from datasets import load_dataset
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)
from transformers import Trainer
from transformers import TrainingArguments
from transformers import ViTFeatureExtractor
from transformers import ViTForImageClassification

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


def preprocess_val(example_batch):
    """Apply val_transforms across a batch."""
    example_batch["pixel_values"] = [val_transforms(image.convert("RGB")) for image in example_batch["image"]]
    origin_labels = example_batch['labels']
    labels_matrix = np.zeros((len(origin_labels), len(labels)))
    # labels_matrix[:, origin_labels] = 1.0
    for idx, label_value in enumerate(origin_labels):
        labels_matrix[idx][label_value] = 1.0
        label_name = labels[label_value]
        if label_name in {'ju_jin_yi', 'song_ya_xuan', 'yu_shu_xin'}:
            labels_matrix[idx][origin_lens + 1] = 1.0

        if label_name in {'ku_luo_mi', 'tan_men_du_zi_lang'}:
            labels_matrix[idx][origin_lens] = 1.0

    example_batch['labels'] = labels_matrix.tolist()
    return example_batch


def preprocess_train(example_batch):
    """Apply train_transforms across a batch."""
    example_batch['pixel_values'] = [train_transforms(image.convert("RGB")) for image in example_batch["image"]]
    origin_labels = example_batch['labels']
    labels_matrix = np.zeros((len(origin_labels), len(labels)))
    # labels_matrix[:, origin_labels] = 1.0
    for idx, label_value in enumerate(origin_labels):
        labels_matrix[idx][label_value] = 1.0
        label_name = labels[label_value]
        if label_name in {'ju_jin_yi', 'song_ya_xuan', 'yu_shu_xin'}:
            labels_matrix[idx][origin_lens + 1] = 1.0

        if label_name in {'ku_luo_mi', 'tan_men_du_zi_lang'}:
            labels_matrix[idx][origin_lens] = 1.0

    example_batch['labels'] = labels_matrix.tolist()
    return example_batch


# source: https://jesusleal.io/2021/04/21/Longformer-multilabel-classification/
def multi_label_metrics(predictions, labels, threshold=0.5):
    # first, apply sigmoid on predictions which are of shape (batch_size, num_labels)
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))
    # next, use threshold to turn them into integer predictions
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1
    # finally, compute metrics
    y_true = labels
    f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
    roc_auc = roc_auc_score(y_true, y_pred, average='micro')
    accuracy = accuracy_score(y_true, y_pred)
    # return as dictionary
    metrics = {'f1': f1_micro_average,
               'roc_auc': roc_auc,
               'accuracy': accuracy}
    return metrics


def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions,
                                           tuple) else p.predictions
    result = multi_label_metrics(
        predictions=preds,
        labels=p.label_ids)
    return result


label_col_name = 'labels'


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
    if label_col_name != 'label':
        ds = ds.rename_column('label', label_col_name)

    labels = ds['train'].features[label_col_name].names
    origin_lens = len(labels)
    labels.append('dong_man') # add extra labels
    labels.append('mingxing') # add extra labels

    train_ds = ds['train']
    val_ds = ds['validation']

    # with_transform
    train_ds.set_transform(preprocess_train)
    val_ds.set_transform(preprocess_val)

    model = ViTForImageClassification.from_pretrained(
        model_name_or_path,
        problem_type="multi_label_classification",
        num_labels=len(labels),
        id2label={str(i): c for i, c in enumerate(labels)},
        label2id={c: str(i) for i, c in enumerate(labels)}
    )

    example = train_ds[0]
    pixel_values = train_ds[0]['pixel_values'].unsqueeze(0)

    output_dir = f"/root/autodl-tmp/output/vit-base-patch16-224-starv4"

    metric_name = "f1"

    training_args = TrainingArguments(
        output_dir=output_dir,
        # per_device_train_batch_size=32,
        per_device_train_batch_size=64,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=4,
        warmup_ratio=0.1,
        fp16=True,
        logging_steps=10,
        learning_rate=2e-4,
        save_total_limit=2,
        remove_unused_columns=False,
        push_to_hub=False,
        report_to=['tensorboard'],
        load_best_model_at_end=True,
        metric_for_best_model=metric_name,
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

    trainer.save_model()
    trainer.log_metrics("train", train_results.metrics)
    trainer.save_metrics("train", train_results.metrics)
    trainer.save_state()

    # We can check with the evaluate method that our Trainer did reload the best model properly (if it was not the last one):
    metrics = trainer.evaluate(val_ds)
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)
