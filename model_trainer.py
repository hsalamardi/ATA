import os
import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    TrainingArguments,
    Trainer,
    EvalPrediction
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import Dataset
import pandas as pd
import logging
import shutil

logger = logging.getLogger(__name__)

class ArabicSentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length)
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

class ModelTrainer:
    def __init__(self, base_path: str = None):
        self.base_path = base_path or os.path.join(Path.home(), '.cache', 'arabic_analyzer')
        self.versions_file = os.path.join(self.base_path, 'model_versions.json')
        os.makedirs(self.base_path, exist_ok=True)
        
        # Base model configuration
        self.base_model_config = {
            'path': 'CAMeL-Lab/bert-base-arabic-camelbert-mix',
            'num_labels': 3,
            'id2label': {0: 'سلبي', 1: 'محايد', 2: 'إيجابي'},
            'label2id': {'سلبي': 0, 'محايد': 1, 'إيجابي': 2}
        }
        
        self.load_versions()
        self._ensure_base_model()

    def _ensure_base_model(self):
        """Ensure base model (version 1) exists"""
        v1_path = self.get_model_path(1)
        if not os.path.exists(v1_path):
            logger.info("Initializing base model (version 1)...")
            # Download and save base model
            tokenizer = BertTokenizer.from_pretrained(self.base_model_config['path'])
            model = BertForSequenceClassification.from_pretrained(
                self.base_model_config['path'],
                num_labels=self.base_model_config['num_labels'],
                id2label=self.base_model_config['id2label'],
                label2id=self.base_model_config['label2id']
            )
            
            # Save model and tokenizer
            os.makedirs(v1_path, exist_ok=True)
            model.save_pretrained(v1_path)
            tokenizer.save_pretrained(v1_path)
            
            # Initialize version history if new
            if not os.path.exists(self.versions_file):
                self.versions = {
                    'sentiment': {
                        'current': 1,
                        'history': [{
                            'version': 1,
                            'timestamp': datetime.now().isoformat(),
                            'metrics': {},
                            'train_samples': 0,
                            'eval_samples': 0,
                            'base_model': self.base_model_config['path']
                        }]
                    }
                }
                self.save_versions()

    def load_versions(self):
        """Load model version history"""
        if os.path.exists(self.versions_file):
            with open(self.versions_file, 'r', encoding='utf-8') as f:
                self.versions = json.load(f)
        else:
            self.versions = {'sentiment': {'current': 1, 'history': []}}
            self.save_versions()

    def save_versions(self):
        """Save model version history"""
        with open(self.versions_file, 'w', encoding='utf-8') as f:
            json.dump(self.versions, f, ensure_ascii=False, indent=2)

    def get_model_path(self, version: int = None):
        """Get path for specific model version"""
        if version is None:
            version = self.versions['sentiment']['current']
        return os.path.join(self.base_path, f'sentiment_v{version}')

    def compute_metrics(self, pred: EvalPrediction):
        """Compute metrics for model evaluation"""
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average='weighted'
        )
        acc = accuracy_score(labels, preds)
        
        return {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }

    def train_sentiment_model(self, train_df: pd.DataFrame, eval_df: pd.DataFrame = None):
        """Train sentiment model with new data"""
        # Validate input data
        required_columns = ['text', 'label']
        if not all(col in train_df.columns for col in required_columns):
            raise ValueError("Training data must contain 'text' and 'label' columns")

        # Map text labels to ids
        train_labels = [self.base_model_config['label2id'][label] for label in train_df['label']]

        # Load current model and tokenizer
        current_version = self.versions['sentiment']['current']
        model_path = self.get_model_path(current_version)
        
        tokenizer = BertTokenizer.from_pretrained(model_path)
        model = BertForSequenceClassification.from_pretrained(
            model_path,
            num_labels=self.base_model_config['num_labels'],
            id2label=self.base_model_config['id2label'],
            label2id=self.base_model_config['label2id']
        )

        # Create datasets
        train_dataset = ArabicSentimentDataset(
            train_df['text'].tolist(),
            train_labels,
            tokenizer
        )

        eval_dataset = None
        if eval_df is not None:
            eval_labels = [self.base_model_config['label2id'][label] for label in eval_df['label']]
            eval_dataset = ArabicSentimentDataset(
                eval_df['text'].tolist(),
                eval_labels,
                tokenizer
            )

        # Training arguments
        training_args = TrainingArguments(
            output_dir=os.path.join(self.base_path, 'results'),
            num_train_epochs=3,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=os.path.join(self.base_path, 'logs'),
            logging_steps=100,
            eval_strategy="epoch" if eval_dataset else "no",
            save_strategy="epoch",
            load_best_model_at_end=True if eval_dataset else False,
        )

        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=self.compute_metrics,
        )

        # Train the model
        train_result = trainer.train()
        metrics = train_result.metrics

        # Evaluate on test set if provided
        if eval_dataset:
            eval_metrics = trainer.evaluate()
            metrics.update(eval_metrics)

        # Save new version
        new_version = current_version + 1
        new_model_path = self.get_model_path(new_version)
        
        # Ensure directory is clean
        if os.path.exists(new_model_path):
            shutil.rmtree(new_model_path)
        os.makedirs(new_model_path)
        
        model.save_pretrained(new_model_path)
        tokenizer.save_pretrained(new_model_path)

        # Update version history
        timestamp = datetime.now().isoformat()
        version_info = {
            'version': new_version,
            'timestamp': timestamp,
            'metrics': metrics,
            'train_samples': len(train_df),
            'eval_samples': len(eval_df) if eval_df is not None else 0
        }
        
        self.versions['sentiment']['history'].append(version_info)
        self.versions['sentiment']['current'] = new_version
        self.save_versions()

        return version_info

    def evaluate_model(self, version: int, eval_df: pd.DataFrame):
        """Evaluate specific model version"""
        # Map text labels to ids
        eval_labels = [self.base_model_config['label2id'][label] for label in eval_df['label']]

        # Load model and tokenizer
        model_path = self.get_model_path(version)
        tokenizer = BertTokenizer.from_pretrained(model_path)
        model = BertForSequenceClassification.from_pretrained(model_path)

        # Create dataset
        eval_dataset = ArabicSentimentDataset(
            eval_df['text'].tolist(),
            eval_labels,
            tokenizer
        )

        # Initialize trainer
        trainer = Trainer(
            model=model,
            compute_metrics=self.compute_metrics,
        )

        # Evaluate
        metrics = trainer.evaluate(eval_dataset)
        return metrics

    def get_version_history(self):
        """Get model version history"""
        return self.versions['sentiment']['history']

    def get_current_version(self):
        """Get current model version"""
        return self.versions['sentiment']['current']
