import torch
import spacy
from torchtext.data.metrics import bleu_score

def translate_sentence(model, sentence, german_vocab, english_vocab, device, max_length=50):
    """
    Translates a German sentence to English using the trained model.
    """
    # Load German tokenizer
    spacy_ger = spacy.load('de_core_news_sm')
    
    # Create tokens using spacy
    if isinstance(sentence, str):
        tokens = [token.text.lower() for token in spacy_ger(sentence)]
    else:
        tokens = [token.lower() for token in sentence]

    # Add <sos> and <eos> tokens
    tokens.insert(0, '<sos>')
    tokens.append('<eos>')

    # Convert tokens to numerical values
    text_to_indices = [german_vocab[token] for token in tokens]

    # Convert to tensor
    sentence_tensor = torch.LongTensor(text_to_indices).unsqueeze(1).to(device)

    # Build encoder hidden, cell state
    with torch.no_grad():
        hidden, cell = model.encoder(sentence_tensor)

    outputs = [english_vocab['<sos>']]

    for _ in range(max_length):
        previous_word = torch.LongTensor([outputs[-1]]).to(device)

        with torch.no_grad():
            output, hidden, cell = model.decoder(previous_word, hidden, cell)
            best_guess = output.argmax(1).item()

        outputs.append(best_guess)

        # If model predicts EOS token, stop generating
        if best_guess == english_vocab['<eos>']:
            break

    translated_sentence = [english_vocab.get_itos()[idx] for idx in outputs]
    # Remove <sos> and <eos> tokens
    return translated_sentence[1:-1]

def bleu(data, model, german_vocab, english_vocab, device):
    """
    Calculate BLEU score for the model's translations.
    """
    targets = []
    outputs = []

    for example in data:
        src = vars(example)['src']
        trg = vars(example)['trg']

        prediction = translate_sentence(model, src, german_vocab, english_vocab, device)
        prediction = prediction[:-1]  # remove <eos> token

        targets.append([trg])
        outputs.append(prediction)

    return bleu_score(outputs, targets)

def save_checkpoint(state, filename="checkpoint.pth.tar"):
    """
    Save model checkpoint to file.
    """
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model, optimizer):
    """
    Load model checkpoint from file.
    """
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
