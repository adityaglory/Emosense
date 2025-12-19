import mlflow
import numpy as np
import torch
import evaluate
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    Trainer, 
    TrainingArguments,
    DataCollatorWithPadding
)
from sklearn.metrics import f1_score, accuracy_score

class MultilabelTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if "labels" in inputs:
            inputs["labels"] = inputs["labels"].float()
            
        return super().compute_loss(model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch)

def main():
    mlflow.set_experiment("EmoSense_GoEmotions_MultiLabel")
    
    print("📥 Loading Google GoEmotions Dataset...")
    dataset = load_dataset("go_emotions", "simplified")
    
    model_ckpt = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
    
    labels_list = [
        "admiration", "amusement", "anger", "annoyance", "approval", "caring", 
        "confusion", "curiosity", "desire", "disappointment", "disapproval", 
        "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief", 
        "joy", "love", "nervousness", "optimism", "pride", "realization", 
        "relief", "remorse", "sadness", "surprise", "neutral"
    ]
    
    id2label = {i: label for i, label in enumerate(labels_list)}
    label2id = {label: i for i, label in enumerate(labels_list)}
    num_labels = len(labels_list)

    def preprocess_function(examples):
        tokenized = tokenizer(examples["text"], truncation=True)
        
        labels_matrix = np.zeros((len(examples["text"]), num_labels))
        
        for idx, labels in enumerate(examples["labels"]):
            for label_id in labels:
                labels_matrix[idx, label_id] = 1.0
        
        tokenized["labels"] = labels_matrix.tolist()
        return tokenized

    print("⚙️ Preprocessing to Multi-Label Vectors...")
    tokenized_ds = dataset.map(preprocess_function, batched=True, remove_columns=dataset["train"].column_names)
    
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        probs = 1 / (1 + np.exp(-predictions))
        y_pred = (probs > 0.5).astype(int)
        
        f1_micro = f1_score(labels, y_pred, average='micro')
        f1_macro = f1_score(labels, y_pred, average='macro')
        acc = accuracy_score(labels, y_pred)
        
        return {
            'f1_micro': f1_micro,
            'f1_macro': f1_macro,
            'accuracy': acc
        }

    model = AutoModelForSequenceClassification.from_pretrained(
        model_ckpt, 
        num_labels=num_labels,
        id2label=id2label, 
        label2id=label2id,
        problem_type="multi_label_classification"
    )

    training_args = TrainingArguments(
        output_dir="./results_multilabel",
        learning_rate=2e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=5,
        weight_decay=0.01,
        eval_strategy="epoch",  
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_micro",
        report_to="none"
    )

    
    trainer = MultilabelTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds["train"].shuffle(seed=42),
        eval_dataset=tokenized_ds["test"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    print("🚀 Starting MULTI-LABEL Training...")
    trainer.train()
    
    print("📊 Evaluating...")
    eval_result = trainer.evaluate()
    print(f"🔥 Final F1-Micro: {eval_result['eval_f1_micro']:.2%}")
    
    print("💾 Saving Model...")
    trainer.save_model("./artifacts/model")
    tokenizer.save_pretrained("./artifacts/model")
    print("✅ Model saved.")

if __name__ == "__main__":
    main()