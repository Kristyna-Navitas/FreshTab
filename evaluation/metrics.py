from collections import Counter

import nltk
import numpy as np
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


def tokenize_sentences(sentences):
    return [nltk.word_tokenize(str(sentence)) for sentence in sentences]


def calculate_self_bleu(sentences, weights):
    scores = []
    sentences = [s for s in sentences if not pd.isna(s)]
    sentences_len = len(sentences)
    sentences = tokenize_sentences(sentences)
    for i, hypothesis in enumerate(sentences):
        if hypothesis:
            references = sentences[:i] + sentences[i+1:]
            scores.append(sentence_bleu(references, hypothesis, weights, SmoothingFunction().method4))
    return np.sum(scores)/sentences_len


def run_self_bleu(list_of_sentences):
    one_nans, two_nans, three_nans, four_nans, five_nans = 0, 0, 0, 0, 0
    one_gram, two_gram, three_gram, four_gram, five_gram = [], [], [], [], []

    for sentences in list_of_sentences:

        score = calculate_self_bleu(sentences, (1.0, 0.0, 0.0, 0.0))
        if pd.isna(score):
            one_nans += 1
        else:
            one_gram.append(score)

        score = calculate_self_bleu(sentences, (0.5, 0.5, 0.0, 0.0))
        if pd.isna(score):
            two_nans += 1
        else:
            two_gram.append(score)

        score = calculate_self_bleu(sentences, (0.33, 0.33, 0.33, 0.0))
        if pd.isna(score):
            three_nans += 1
        else:
            three_gram.append(score)

        score = calculate_self_bleu(sentences, (0.25, 0.25, 0.25, 0.25))
        if pd.isna(score):
            four_nans += 1
        else:
            four_gram.append(score)

        score = calculate_self_bleu(sentences, (0.2, 0.2, 0.2, 0.2, 0.2))
        if pd.isna(score):
            five_nans += 1
        else:
            five_gram.append(score)
            #if score > 0.4:
            #   for s in sentences:
            #      print(s)
            #  print()

    return {'BLEU1': sum(one_gram)/(len(one_gram)-one_nans), 'BLEU2': sum(two_gram)/(len(two_gram)-two_nans),
            'BLEU3': sum(three_gram)/(len(three_gram)-three_nans), 'BLEU4': sum(four_gram)/(len(four_gram)-four_nans),
            'BLEU5': sum(five_gram)/(len(five_gram)-five_nans)}


def get_avg_length(list_of_sentences):
    length = 0
    for sentences in list_of_sentences:
        for sentence in sentences:
            length += len(str(sentence))
    return length/(len(list_of_sentences)*5)


def get_unique_tokens(list_of_sentences):
    unique_tokens_per_table_sentences = 0
    for sentences in list_of_sentences:
        tokens = tokenize_sentences(sentences)
        flat_tokens = [item for sublist in tokens for item in sublist]
        unique_tokens_per_table_sentences += len(set(flat_tokens))
    return unique_tokens_per_table_sentences/len(list_of_sentences)


def get_shannon_entropy(list_of_sentences):
    sum_entropy = 0
    for sentences in list_of_sentences:
        tokens = tokenize_sentences(sentences)
        flat_tokens = [item for sublist in tokens for item in sublist]
        counts = Counter(flat_tokens)
        probs = np.array(list(counts.values())) / sum(counts.values())
        sum_entropy = -np.sum(probs * np.log2(probs))

    return sum_entropy


def get_msttr(list_of_sentences):
    sum_msttr = 0
    for sentences in list_of_sentences:
        tokens = tokenize_sentences(sentences)
        msttr = 0
        for sentence in tokens:  # ttr
            msttr += len(set(sentence))/len(sentence)
        msttr = msttr/len(tokens)
        sum_msttr += msttr
    return sum_msttr/len(list_of_sentences)

