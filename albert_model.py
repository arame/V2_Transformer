from transformers import AlbertForSequenceClassification, AlbertTokenizer, AlbertConfig
from config import Hyper
from helper import Helper

def load_albert_model():
    # Load AlbertForSequenceClassification, the pretrained ALBERT model with a single 
    # linear classification layer on top. 
    Helper.printline(f'Loading ALBERT model using the {Hyper.model_name} modal...')
    _config = set_dropout()
    model = AlbertForSequenceClassification.from_pretrained(
        Hyper.model_name,               # Use the 12-layer ALBERT model, with an uncased vocab.
        config = _config
    )

    return model

def set_dropout():
    config = AlbertConfig()
    config.num_labels = Hyper.num_labels    # Labels are either positive or negative sentiment.
    config.attention_probs_dropout_prob = Hyper.dropout_rate
    config.hidden_dropout_prob = Hyper.dropout_rate
    config.output_attentions = False        # Do not return attentions weights.
    config.output_hidden_states = False     # Do not return all hidden-states.
    return config

def load_albert_tokeniser():
    # Load the ALBERT tokenizer.
    Helper.printline(f'Loading ALBERT tokenizer using the {Hyper.model_name} modal...')
    tokenizer = AlbertTokenizer.from_pretrained(Hyper.model_name)
    return tokenizer