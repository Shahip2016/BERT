
import os
import torch
from torch.utils.data import DataLoader
from .config import BertConfig
from .bert_heads import BertForPreTraining
from .dataset import BertDataset, BertTokenizer

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    config = BertConfig()
    model = BertForPreTraining(config)
    model.to(device)
    model.train()

    tokenizer = BertTokenizer()
    dataset = BertDataset(args.corpus_path, tokenizer)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    print("Starting training...")
    for epoch in range(args.epochs):
        for i, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            segment_ids = batch["segment_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            masked_lm_labels = batch["masked_lm_labels"].to(device)
            next_sentence_label = batch["next_sentence_label"].to(device)

            optimizer.zero_grad()
            prediction_scores, seq_relationship_score, loss = model(
                input_ids, segment_ids, attention_mask, masked_lm_labels, next_sentence_label
            )

            loss.backward()
            optimizer.step()

            if i % 10 == 0:
                print(f"Epoch: {epoch}, Step: {i}, Loss: {loss.item()}")

    print("Training complete.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus_path", type=str, default="corpus.txt", help="Path to text corpus")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    args = parser.parse_args()
    
    # Create a dummy corpus if not exists for testing
    if not os.path.exists(args.corpus_path):
        with open(args.corpus_path, "w", encoding="utf-8") as f:
            f.write("The quick brown fox jumps over the lazy dog.\n")
            f.write("The dog is lazy but the fox is quick.\n")
            f.write("BERT is a powerful model for NLP.\n")
            f.write("Natural Language Processing is fun.\n")

    train(args)
