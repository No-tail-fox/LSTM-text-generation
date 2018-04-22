import os
import pickle


def load_text(path):
    input_file=os.path.join(path)

    with open(input_file,'r',encoding="utf-8-sig") as file:
        input_text=file.read()
    return input_text


def preprocess_and_save_data(text, token_lookup, create_lookup_tables):
    for key, token in token_lookup.items():
        text = text.replace(key, '{}'.format(token))
    text = list(text)
    vocab_to_int,int_to_vocab=create_lookup_tables(text)
    int_text = [vocab_to_int[word] for word in text]
    pickle.dump((int_text, vocab_to_int, int_to_vocab, token_lookup), open('preprocess.p', 'wb'))


def load_preprocess():
    return pickle.load(open('preprocess.p', mode='rb'))