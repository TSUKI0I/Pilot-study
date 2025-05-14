import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
import seaborn as sns
import random
import warnings

warnings.filterwarnings('ignore')

# Set random seeds to ensure the reproducibility of the results
def set_seed(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)


set_seed()


# Data enhancement class
class DataAugmenter:
    def __init__(self, tokenizer, device='cpu'):
        self.tokenizer = tokenizer
        self.device = device

    def random_insertion(self, text, p=0.1):
        words = text.split()
        num_new_words = int(len(words) * p)
        for _ in range(num_new_words):
            word = random.choice(words)
            idx = random.randint(0, len(words) - 1)
            words.insert(idx, word)
        return ' '.join(words)

    def random_deletion(self, text, p=0.1):
        words = text.split()
        if len(words) == 1:
            return words[0]
        new_words = []
        for word in words:
            if random.uniform(0, 1) > p:
                new_words.append(word)
        if len(new_words) == 0:
            return random.choice(words)
        return ' '.join(new_words)

    def random_swap(self, text, p=0.1):
        words = text.split()
        num_swaps = int(len(words) * p)
        for _ in range(num_swaps):
            idx1, idx2 = random.sample(range(len(words)), 2)
            words[idx1], words[idx2] = words[idx2], words[idx1]
        return ' '.join(words)

    def augment(self, text):
        augmentations = [
            lambda x: self.random_insertion(x),
            lambda x: self.random_deletion(x),
            lambda x: self.random_swap(x)
        ]
        augmentation = random.choice(augmentations)
        return augmentation(text)


# Custom dataset class
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len, augmenter=None, augment_ratio=0.2):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.augmenter = augmenter
        self.augment_ratio = augment_ratio

        # data enhancement
        if self.augmenter:
            original_size = len(self.texts)
            augment_size = int(original_size * self.augment_ratio)
            indices = np.random.choice(original_size, augment_size)

            augmented_texts = []
            augmented_labels = []

            for idx in indices:
                text = self.texts[idx]
                label = self.labels[idx]
                augmented_text = self.augmenter.augment(text)

                augmented_texts.append(augmented_text)
                augmented_labels.append(label)

            self.texts = self.texts + augmented_texts
            self.labels = self.labels + augmented_labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }


# BERT classification mode
class BertClassifier(nn.Module):
    def __init__(self, n_classes, dropout=0.3, bert_model_name='bert-base-uncased'):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False
        )
        output = self.dropout(pooled_output)
        return self.classifier(output)


# Training function
def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler, n_examples):
    model = model.train()
    losses = []
    correct_predictions = 0

    for d in tqdm(data_loader):
        input_ids = d['input_ids'].to(device)
        attention_mask = d['attention_mask'].to(device)
        labels = d['label'].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, labels)

        correct_predictions += torch.sum(preds == labels)
        losses.append(loss.item())

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return correct_predictions.double() / n_examples, np.mean(losses)


# evaluation function
def eval_model(model, data_loader, loss_fn, device, n_examples):
    model = model.eval()
    losses = []
    correct_predictions = 0

    with torch.no_grad():
        for d in tqdm(data_loader):
            input_ids = d['input_ids'].to(device)
            attention_mask = d['attention_mask'].to(device)
            labels = d['label'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            _, preds = torch.max(outputs, dim=1)
            loss = loss_fn(outputs, labels)

            correct_predictions += torch.sum(preds == labels)
            losses.append(loss.item())

    return correct_predictions.double() / n_examples, np.mean(losses)


# forecasting function
def predict(model, text, tokenizer, max_len, device):
    model.eval()

    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_len,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        output = model(input_ids, attention_mask)
        _, prediction = torch.max(output, dim=1)

    return prediction.item()


# Save the training log
def save_training_log(train_acc, train_loss, val_acc, val_loss, epochs, save_path='training_log.csv'):
    log_df = pd.DataFrame({
        'Epoch': range(1, epochs + 1),
        'Train Accuracy': train_acc,
        'Train Loss': train_loss,
        'Validation Accuracy': val_acc,
        'Validation Loss': val_loss
    })
    log_df.to_csv(save_path, index=False)
    print(f"{save_path}")
    return log_df


# 可视化训练过程
def visualize_training(log_df, save_path='training_plots.png'):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(log_df['Epoch'], log_df['Train Accuracy'], label='Training accuracy')
    plt.plot(log_df['Epoch'], log_df['Validation Accuracy'], label='Verifying accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('accuracy')
    plt.legend()
    plt.title('Training and verifying accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(log_df['Epoch'], log_df['Train Loss'], label='Training loss')
    plt.plot(log_df['Epoch'], log_df['Validation Loss'], label='Verifying loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and verifying loss')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# Visual performance indicators
def visualize_metrics(y_true, y_pred, class_names, save_path='metrics_report.png'):
    plt.figure(figsize=(15, 5))

    # confusion matrix
    plt.subplot(1, 2, 1)
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('predict')
    plt.ylabel('ture')
    plt.title('confusion matrix')

    plt.subplot(1, 2, 2)
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df = report_df.drop(['accuracy', 'macro avg', 'weighted avg'])
    sns.heatmap(report_df[['precision', 'recall', 'f1-score']], annot=True, cmap='YlGnBu', fmt='.2f')
    plt.title('report')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f" {save_path}")


def main():
    config = {
        'bert_model_name': 'bert-base-uncased',
        'max_len': 128,
        'batch_size': 16,
        'epochs': 50,
        'learning_rate': 2e-5,
        'dropout': 0.3,
        'augment_ratio': 0.2,
        'n_classes': 2,
        'class_names': ['0', '1'],
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }

    print(f"device: {config['device']}")

    # Create a save directory
    if not os.path.exists('models'):
        os.makedirs('models')
    if not os.path.exists('plots'):
        os.makedirs('plots')


    data = pd.read_csv("ai-ga-dataset.csv")
    texts = data["abstract"].tolist()
    labels = data["label"].tolist()
    idx = list(range(len(texts)))
    random.shuffle(idx)
    idx = idx[:3000]
    texts = [texts[_] for _ in idx]
    labels = [labels[_] for _ in idx]

    # Divide the training set and the validation set
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )

    # Loading tokenizer
    print("Loading tokenizer...")
    tokenizer = BertTokenizer.from_pretrained(config['bert_model_name'])

    # Data enhancer
    augmenter = DataAugmenter(tokenizer, config['device'])

    # Create datasets and data loaders
    train_dataset = TextDataset(
        train_texts,
        train_labels,
        tokenizer,
        config['max_len'],
        augmenter=augmenter,
        augment_ratio=config['augment_ratio']
    )

    val_dataset = TextDataset(
        val_texts,
        val_labels,
        tokenizer,
        config['max_len']
    )

    train_data_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4
    )

    val_data_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        num_workers=4
    )

    # Initialize the model
    model = BertClassifier(
        n_classes=config['n_classes'],
        dropout=config['dropout'],
        bert_model_name=config['bert_model_name']
    ).to(config['device'])

    # Optimizer and loss function
    optimizer = AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        correct_bias=False
    )

    total_steps = len(train_data_loader) * config['epochs']

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    loss_fn = nn.CrossEntropyLoss().to(config['device'])

    # The Training Cycle
    history = {
        'train_acc': [],
        'train_loss': [],
        'val_acc': [],
        'val_loss': []
    }

    best_accuracy = 0

    for epoch in range(config['epochs']):
        print(f'Epoch {epoch + 1}/{config["epochs"]}')
        print('-' * 10)

        train_acc, train_loss = train_epoch(
            model,
            train_data_loader,
            loss_fn,
            optimizer,
            config['device'],
            scheduler,
            len(train_dataset)
        )

        print(f'Train loss: {train_loss:.4f}, Train accuracy: {train_acc:.4f}')

        val_acc, val_loss = eval_model(
            model,
            val_data_loader,
            loss_fn,
            config['device'],
            len(val_dataset)
        )

        print(f'Val loss: {val_loss:.4f}, Val accuracy: {val_acc:.4f}')
        print()

        history['train_acc'].append(train_acc.item())
        history['train_loss'].append(train_loss)
        history['val_acc'].append(val_acc.item())
        history['val_loss'].append(val_loss)

        if val_acc > best_accuracy:
            torch.save(model.state_dict(), 'models/best_model.bin')
            best_accuracy = val_acc
            print("The best model has been saved")

    log_df = save_training_log(
        history['train_acc'],
        history['train_loss'],
        history['val_acc'],
        history['val_loss'],
        config['epochs'],
        save_path='plots/training_log.csv'
    )

    #
    # visualize_training(log_df, save_path='plots/training_plots.png')

    # Load the best model for evaluation
    model = BertClassifier(n_classes=config['n_classes'])
    model.load_state_dict(torch.load('models/best_model.bin'))
    model = model.to(config['device'])

    # Obtain the prediction results of the validation set
    y_pred = []
    y_true = []

    model.eval()
    with torch.no_grad():
        for d in val_data_loader:
            input_ids = d['input_ids'].to(config['device'])
            attention_mask = d['attention_mask'].to(config['device'])
            labels = d['label'].to(config['device'])

            outputs = model(input_ids, attention_mask)
            _, preds = torch.max(outputs, dim=1)

            y_pred.extend(preds.cpu().tolist())
            y_true.extend(labels.cpu().tolist())

    # Calculate performance indicators
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')

    print(f'Finally verify the accuracy rate of the set: {accuracy:.4f}')
    print(f'Final verification set F1 score: {f1:.4f}')

    # Visual performance indicators
    visualize_metrics(y_true, y_pred, config['class_names'], save_path='plots/metrics_report.png')

    # Example prediction
    sample_text = "Rift Valley fever virus (RVFV) (genus Phlebovirus, family Bunyaviridae) is a negative-stranded RNA virus with a tripartite genome. RVFV is transmitted by mosquitoes and causes fever and severe hemorrhagic illness among humans, and fever and high rates of abortions in livestock. A nonstructural RVFV NSs protein inhibits the transcription of host mRNAs, including interferon-尾 mRNA, and is a major virulence factor. The present study explored a novel function of the RVFV NSs protein by testing the replication of RVFV lacking the NSs gene in the presence of actinomycin D (ActD) or 伪-amanitin, both of which served as a surrogate of the host mRNA synthesis suppression function of the NSs. In the presence of the host-transcriptional inhibitors, the replication of RVFV lacking the NSs protein, but not that carrying NSs, induced double-stranded RNA-dependent protein kinase (PKR)鈥搈ediated eukaryotic initiation factor (eIF)2伪 phosphorylation, leading to the suppression of host and viral protein translation. RVFV NSs promoted post-transcriptional downregulation of PKR early in the course of the infection and suppressed the phosphorylated eIF2伪 accumulation. These data suggested that a combination of RVFV replication and NSs-induced host transcriptional suppression induces PKR-mediated eIF2伪 phosphorylation, while the NSs facilitates efficient viral translation by downregulating PKR and inhibiting PKR-mediated eIF2伪 phosphorylation. Thus, the two distinct functions of the NSs, i.e., the suppression of host transcription, including that of type I interferon mRNAs, and the downregulation of PKR, work together to prevent host innate antiviral functions, allowing efficient replication and survival of RVFV in infected mammalian hosts."
    prediction = predict(model, sample_text, tokenizer, config['max_len'], config['device'])
    print(f'{sample_text}')
    print(f'{config["class_names"][prediction]}')


if __name__ == "__main__":
    main()
