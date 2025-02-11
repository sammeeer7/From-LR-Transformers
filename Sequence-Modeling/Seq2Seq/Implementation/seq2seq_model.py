import torch
import torch.nn as nn
import torch.optim as optim
import random
import spacy
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.utils import get_tokenizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils import translate_sentence, bleu, save_checkpoint, load_checkpoint
from torchtext.datasets import Multi30k
from typing import Iterable, List

# Load spaCy tokenizers
spacy_ger = spacy.load('de_core_news_sm')
spacy_eng = spacy.load('en_core_web_sm')

def tokenizer_ger(text):
    return [tok.text.lower() for tok in spacy_ger.tokenizer(text)]

def tokenizer_eng(text):
    return [tok.text.lower() for tok in spacy_eng.tokenizer(text)]

# Create the tokenizers
german_tokenizer = get_tokenizer(tokenizer_ger)
english_tokenizer = get_tokenizer(tokenizer_eng)

# Helper function to yield tokens
def yield_tokens(data_iter: Iterable, language: str) -> List[str]:
    language_index = 0 if language == 'de' else 1
    for data_sample in data_iter:
        yield tokenizer_ger(data_sample[language_index]) if language == 'de' else tokenizer_eng(data_sample[language_index])

# Load the Multi30k dataset
train_iter, valid_iter, test_iter = Multi30k(split=('train', 'valid', 'test'))

# Create vocabulary
special_symbols = ['<unk>', '<pad>', '<sos>', '<eos>']
german_vocab = build_vocab_from_iterator(
    yield_tokens(train_iter, 'de'),
    min_freq=2,
    specials=special_symbols,
    special_first=True
)
german_vocab.set_default_index(german_vocab['<unk>'])

# Need to reload the iterator because it was consumed
train_iter = Multi30k(split='train')

english_vocab = build_vocab_from_iterator(
    yield_tokens(train_iter, 'en'),
    min_freq=2,
    specials=special_symbols,
    special_first=True
)
english_vocab.set_default_index(english_vocab['<unk>'])

# Reload iterator again for training
train_iter = Multi30k(split='train')
valid_iter = Multi30k(split='valid')
test_iter = Multi30k(split='test')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define Hyperparameters
num_epochs = 20
learning_rate = 0.001
batch_size = 64
input_size_encoder = len(german_vocab)
input_size_decoder = len(english_vocab)
output_size = len(english_vocab)
encoder_embedding_size = 300
decoder_embedding_size = 300
hidden_size = 1024
num_layers = 2
enc_dropout = 0.5
dec_dropout = 0.5

# Define Encoder
class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, dropout):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=dropout, batch_first=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: (seq_length, batch_size)
        embedding = self.dropout(self.embedding(x))
        # embedding shape: (seq_length, batch_size, embedding_size)
        
        outputs, (hidden, cell) = self.rnn(embedding)
        return hidden, cell

# Define Decoder
class Decoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, output_size, num_layers, dropout):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=dropout, batch_first=False)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, hidden, cell):
        # x shape: (batch_size)
        x = x.unsqueeze(0)
        # x shape: (1, batch_size)
        
        embedding = self.dropout(self.embedding(x))
        # embedding shape: (1, batch_size, embedding_size)
        
        outputs, (hidden, cell) = self.rnn(embedding, (hidden, cell))
        # outputs shape: (1, batch_size, hidden_size)
        
        predictions = self.fc(outputs)
        # predictions shape: (1, batch_size, output_size)
        predictions = predictions.squeeze(0)
        return predictions, hidden, cell

# Seq2Seq Model
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, source, target, teacher_force_ratio=0.5):
        batch_size = source.shape[1]
        target_len = target.shape[0]
        target_vocab_size = len(english_vocab)
        
        outputs = torch.zeros(target_len, batch_size, target_vocab_size).to(device)
        
        hidden, cell = self.encoder(source)
        
        # First input to the decoder is the <sos> token
        x = target[0]
        
        for t in range(1, target_len):
            output, hidden, cell = self.decoder(x, hidden, cell)
            outputs[t] = output
            
            # Get the best word the Decoder predicted (index in the vocabulary)
            best_guess = output.argmax(1)
            
            # With probability of teacher_force_ratio we take the actual next word
            # otherwise we take the word that the Decoder predicted it to be.
            # Teacher Forcing is used more at the beginning of training
            x = target[t] if random.random() < teacher_force_ratio else best_guess

        return outputs

# Define data processing functions
def collate_fn(batch):
    src_batch, trg_batch = [], []
    for src_sample, trg_sample in batch:
        src_batch.append(torch.tensor([german_vocab[token] for token in tokenizer_ger(src_sample)], dtype=torch.long))
        trg_batch.append(torch.tensor([english_vocab[token] for token in tokenizer_eng(trg_sample)], dtype=torch.long))
    
    src_batch = nn.utils.rnn.pad_sequence(src_batch, padding_value=german_vocab['<pad>'])
    trg_batch = nn.utils.rnn.pad_sequence(trg_batch, padding_value=english_vocab['<pad>'])
    
    return src_batch, trg_batch

# Create data loaders
train_dataloader = DataLoader(
    list(train_iter), 
    batch_size=batch_size, 
    collate_fn=collate_fn,
    shuffle=True
)
valid_dataloader = DataLoader(
    list(valid_iter), 
    batch_size=batch_size, 
    collate_fn=collate_fn
)
test_dataloader = DataLoader(
    list(test_iter), 
    batch_size=batch_size, 
    collate_fn=collate_fn
)

# Initialize Model
encoder_net = Encoder(
    input_size_encoder,
    encoder_embedding_size,
    hidden_size,
    num_layers,
    enc_dropout
).to(device)

decoder_net = Decoder(
    input_size_decoder,
    decoder_embedding_size,
    hidden_size,
    output_size,
    num_layers,
    dec_dropout
).to(device)

model = Seq2Seq(encoder_net, decoder_net).to(device)

# Loss and Optimizer
pad_idx = english_vocab['<pad>']
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# TensorBoard
writer = SummaryWriter(f"runs/loss_plot")
step = 0

# Training Loop
for epoch in range(num_epochs):
    print(f'Epoch [{epoch + 1}/{num_epochs}]')
    model.train()
    
    for batch_idx, (src, trg) in enumerate(train_dataloader):
        src = src.to(device)
        trg = trg.to(device)
        
        output = model(src, trg)
        
        output = output[1:].reshape(-1, output.shape[2])
        trg = trg[1:].reshape(-1)
        
        optimizer.zero_grad()
        loss = criterion(output, trg)
        loss.backward()
        
        # Clip gradients to avoid exploding gradient problems
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        
        optimizer.step()
        
        # Log the loss
        writer.add_scalar('Training Loss', loss.item(), global_step=step)
        step += 1
        
        if batch_idx % 100 == 0:
            print(f'Batch [{batch_idx}/{len(train_dataloader)}] Loss: {loss.item():.4f}')
    
    # Validation step
    model.eval()
    valid_losses = []
    with torch.no_grad():
        for src, trg in valid_dataloader:
            src = src.to(device)
            trg = trg.to(device)
            
            output = model(src, trg, teacher_force_ratio=0)
            output = output[1:].reshape(-1, output.shape[2])
            trg = trg[1:].reshape(-1)
            
            loss = criterion(output, trg)
            valid_losses.append(loss.item())
    
    valid_loss = sum(valid_losses) / len(valid_losses)
    print(f'Validation Loss: {valid_loss:.4f}')
    writer.add_scalar('Validation Loss', valid_loss, global_step=epoch)
    
    # Save checkpoint
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    save_checkpoint(checkpoint, filename=f"checkpoint_epoch_{epoch+1}.pth.tar")

writer.close()

# Test the model
model.eval()
test_losses = []
with torch.no_grad():
    for src, trg in test_dataloader:
        src = src.to(device)
        trg = trg.to(device)
        
        output = model(src, trg, teacher_force_ratio=0)
        output = output[1:].reshape(-1, output.shape[2])
        trg = trg[1:].reshape(-1)
        
        loss = criterion(output, trg)
        test_losses.append(loss.item())

test_loss = sum(test_losses) / len(test_losses)
print(f'Test Loss: {test_loss:.4f}')

# Try a translation
example_idx = 0
src, trg = next(iter(test_dataloader))
src = src.to(device)
with torch.no_grad():
    translated_sentence = translate_sentence(
        model, src[:, example_idx].cpu().numpy(),
        german_vocab, english_vocab, device
    )
print("Example translation:")
print(f"Predicted: {' '.join(translated_sentence)}")
