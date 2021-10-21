from transformers import RobertaForSequenceClassification, RobertaTokenizer, RobertaConfig
from config import Hyper
from helper import Helper

def load_roberta_model():
    # Load RobertaForSequenceClassification, the pretrained ROBERTA model with a single 
    # linear classification layer on top. 
    Helper.printline(f'Loading ROBERTA model using the {Hyper.model_name} modal...')
    _config = set_dropout()
    model = RobertaForSequenceClassification.from_pretrained(
        Hyper.model_name,               # Use the 12-layer ROBERTA model, with an uncased vocab.
        config = _config
    )

    return model

def set_dropout():
    config = RobertaConfig()
    config.num_labels = Hyper.num_labels    # Labels are either positive or negative sentiment.
    config.attention_probs_dropout_prob = Hyper.dropout_rate
    config.hidden_dropout_prob = Hyper.dropout_rate
    config.output_attentions = False        # Do not return attentions weights.
    config.output_hidden_states = False     # Do not return all hidden-states.
    return config

def load_roberta_tokeniser():
    # Load the ROBERTA tokenizer.
    Helper.printline(f'Loading ROBERTA tokenizer using the {Hyper.model_name} modal...')
    tokenizer = RobertaTokenizer.from_pretrained(Hyper.model_name)
    return tokenizer