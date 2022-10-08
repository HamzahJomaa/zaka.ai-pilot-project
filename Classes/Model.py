import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from Functions.preprocessing import clean_text
import numpy as np

class Model:
    def __init__(self, tokenizer_folder, inference_folder, max_len_en, max_len_fr):
        self.tokenizer_folder = tokenizer_folder
        self.inference_folder = inference_folder
        self.en_tokenizer = None
        self.fr_tokenizer = None
        self.encoder = None
        self.decoder = None
        self.max_len_en = max_len_en
        self.max_len_fr = max_len_fr

    def load_files(self):
        self.en_tokenizer = pickle.load(open("{}/en.pickle".format(self.tokenizer_folder), "rb"))
        self.fr_tokenizer = pickle.load(open("{}/fr.pickle".format(self.tokenizer_folder), "rb"))
        self.encoder = load_model("{}/encoder.h5".format(self.inference_folder))
        self.decoder = load_model("{}/decoder.h5".format(self.inference_folder))

    def prepocess(self, text):
        cleaned_text = clean_text(text)
        tokenized_text = self.en_tokenizer.texts_to_sequences([cleaned_text])
        padded_text = pad_sequences(tokenized_text,maxlen=self.max_len_en)
        return padded_text

    def translate_to_english(self, input_seq):
        # Encode the input as state vectors.
        enc_output, enc_h, enc_c = self.encoder.predict(input_seq, verbose=0)

        # Generate empty target sequence of length 1.
        target_seq = np.zeros((1, 1))
        seq = []
        # Sampling loop for a batch of sequences
        # (to simplify, here we assume a batch of size 1).
        stop_condition = False
        decoded_sentence = ''

        while not stop_condition:
            output_tokens, h, c = self.decoder.predict([target_seq] + [enc_output, enc_h, enc_c], verbose=0)
            # Sample a token
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            seq.append(sampled_token_index)
            # convert max index number to marathi word
            sampled_char = ""
            if (sampled_token_index != 0):
                sampled_char = self.fr_tokenizer.index_word[sampled_token_index]
            # aapend it to decoded sent
            decoded_sentence += ' ' + sampled_char

            # Exit condition: either hit max length or find stop token.
            if (sampled_char == 'eos' or len(decoded_sentence.split()) >= 24):
                stop_condition = True

            # Update the target sequence (of length 1).
            target_seq = np.zeros((1, 1))
            target_seq[0, 0] = sampled_token_index

            # Update states
            enc_h, enc_c = h, c

        sentence = decoded_sentence.split(" ")
        return " ".join([i for i in sentence if i])


        