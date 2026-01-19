
import torch
from .config import BertConfig
from .bert_heads import BertForPreTraining
from .dataset import BertTokenizer

def verify():
    print("Verifying BERT setup...")
    
    # 1. Config and Model
    config = BertConfig(
        vocab_size=100,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=32
    )
    model = BertForPreTraining(config)
    print("Model initialized.")

    # 2. Dummy Data
    tokenizer = BertTokenizer() # Using default vocab which is larger than 100, but we will limit tokens
    
    ids = torch.tensor([[2, 10, 11, 3, 12, 13, 3] + [0]*25]) # [CLS] ... [SEP] ... [SEP] [PAD]...
    segment_ids = torch.tensor([[0, 0, 0, 0, 1, 1, 1] + [0]*25])
    mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1] + [0]*25])
    
    # Ensure ids are within vocab_size
    ids = torch.clamp(ids, 0, 99)
    
    mlm_labels = torch.full((1, 32), -1, dtype=torch.long)
    mlm_labels[0, 1] = 10 # Mask the second token
    nsp_label = torch.tensor([0], dtype=torch.long) # IsNext

    print("Dummy data prepared.")

    # 3. Forward Pass
    prediction_scores, seq_relationship_score, total_loss = model(
        input_ids=ids,
        token_type_ids=segment_ids,
        attention_mask=mask,
        masked_lm_labels=mlm_labels,
        next_sentence_label=nsp_label
    )
    
    print(f"Prediction scores shape: {prediction_scores.shape}") # Should be [1, 32, 100]
    print(f"NSP score shape: {seq_relationship_score.shape}") # Should be [1, 2]
    print(f"Total Loss: {total_loss.item()}")

    assert prediction_scores.shape == (1, 32, 100)
    assert seq_relationship_score.shape == (1, 2)
    assert total_loss.item() > 0

    # 4. Backward Pass (Check gradients)
    total_loss.backward()
    
    has_grad = False
    for param in model.parameters():
        if param.grad is not None:
            has_grad = True
            break
            
    if has_grad:
        print("Gradients computed successfully.")
    else:
        print("ERROR: No gradients computed.")

    print("Verification complete!")

if __name__ == "__main__":
    verify()
