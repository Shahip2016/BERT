
import random
import torch
from torch.utils.data import Dataset

class BertTokenizer:
    # A simple whitespace tokenizer for demonstration purposes.
    # In a real scenario, use a proper WordPiece tokenizer.
    def __init__(self, vocab_file=None):
        self.vocab = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, "[MASK]": 4}
        # Add some dummy vocabulary
        words = "the of and to a in for is on that by this with i you it not or be are from at as your all have new more an was we will home can us about if page my has search free but our one other do no information time they site he up may what which their news out use any there see only so his when contact here business who web also now help get pm view online first am been would how were me services some these click like service x than find price date back top people had list name just over state year day into email two health n world re next used go b work last most products music buy data make them should product system post less fly city best set three available policy copyright support same doing".split()
        for i, w in enumerate(words):
            self.vocab[w] = i + 5
        self.ids_to_tokens = {v: k for k, v in self.vocab.items()}

    def tokenize(self, text):
        return [self.vocab.get(w, self.vocab["[UNK]"]) for w in text.split()]

    def convert_tokens_to_ids(self, tokens):
        return tokens

    @property
    def vocab_size(self):
        return len(self.vocab)

class BertDataset(Dataset):
    def __init__(self, corpus_path, tokenizer, seq_len=128):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.sentences = []
        
        # Read corpus - treating each line as a sentence for simplicity
        with open(corpus_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip().lower()
                if line:
                    self.sentences.append(line)
                    
    def __len__(self):
        return len(self.sentences) - 1

    def __getitem__(self, idx):
        # Generate NSP pair
        t1 = self.sentences[idx]
        
        if random.random() > 0.5:
            # IsNext
            t2 = self.sentences[idx + 1]
            is_next_label = 1 # 1 means IsNext? Usually 0 is IsNext in BERT implementation but let's define 1 as True. 
            # Wait, BERT paper: "class label 0 for IsNext and 1 for NotNext". Let's stick to that.
            is_next_label = 0
        else:
            # NotNext
            t2 = self.sentences[random.randint(0, len(self.sentences) - 1)]
            is_next_label = 1

        t1_ids = self.tokenizer.tokenize(t1)
        t2_ids = self.tokenizer.tokenize(t2)

        # Truncate to ensure fit [CLS] t1 [SEP] t2 [SEP]
        max_len = self.seq_len - 3
        while len(t1_ids) + len(t2_ids) > max_len:
            if len(t1_ids) > len(t2_ids):
                t1_ids.pop()
            else:
                t2_ids.pop()

        input_ids = [self.tokenizer.vocab["[CLS]"]] + t1_ids + [self.tokenizer.vocab["[SEP]"]] + t2_ids + [self.tokenizer.vocab["[SEP]"]]
        segment_ids = [0] * (len(t1_ids) + 2) + [1] * (len(t2_ids) + 1)
        
        # Masking
        masked_ids, masked_lm_labels = self.mask_tokens(input_ids)

        # Padding
        padding_len = self.seq_len - len(input_ids)
        masked_ids += [self.tokenizer.vocab["[PAD]"]] * padding_len
        segment_ids += [0] * padding_len
        masked_lm_labels += [-1] * padding_len # -1 means ignore loss
        
        # specific for attention mask (1 for real tokens, 0 for pad)
        attention_mask = [1] * len(input_ids) + [0] * padding_len

        return {
            "input_ids": torch.tensor(masked_ids),
            "segment_ids": torch.tensor(segment_ids),
            "attention_mask": torch.tensor(attention_mask),
            "masked_lm_labels": torch.tensor(masked_lm_labels),
            "next_sentence_label": torch.tensor(is_next_label)
        }

    def mask_tokens(self, input_ids):
        """Prepare masked tokens inputs/labels for masked language modeling."""
        labels = []
        masked_ids = []
        for token_id in input_ids:
            if token_id in [self.tokenizer.vocab["[CLS]"], self.tokenizer.vocab["[SEP]"], self.tokenizer.vocab["[PAD]"]]:
                masked_ids.append(token_id)
                labels.append(-1)
                continue
            
            prob = random.random()
            if prob < 0.15:
                prob /= 0.15
                if prob < 0.8:
                    masked_ids.append(self.tokenizer.vocab["[MASK]"])
                elif prob < 0.9:
                    masked_ids.append(random.randint(5, len(self.tokenizer.vocab)-1))
                else:
                    masked_ids.append(token_id)
                labels.append(token_id)
            else:
                masked_ids.append(token_id)
                labels.append(-1)
        return masked_ids, labels
