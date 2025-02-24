import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from string import punctuation
from collections import Counter
def padding_features(reviews_int, sequence_length):

    features = np.zeros((len(reviews_int), sequence_length), dtype = int)

    for i, review in enumerate(reviews_int):
        review_len = len(review)

        if review_len <= sequence_length:
            features[i, (sequence_length - review_len) : ] = np.array(review)
        else:
            features[i, :] = np.array(review[ : sequence_length])
    
    return features

def data_preprocess(file_path):

    data = pd.read_csv(file_path)

    enocoder = LabelEncoder()
    data["sentiment"] = enocoder.fit_transform(data["sentiment"])

    data["review"] = data["review"].apply(lambda x : x.lower())
    data["review"] = data["review"].apply(lambda x: ''.join([c for c in x if c not in punctuation]))
    data["len_review"] = data["review"].apply(lambda x : len(x))

    all_text = data["review"].tolist()

    all_text = ' '.join(all_text)

    words = all_text.split()
    count_words = Counter(words)

    total_words = len(words)
    sorted_words = count_words.most_common(total_words)

    vocab_to_int = {w : i+1 for i, (w,c) in enumerate(sorted_words)}

    reviews_int = []
    for review in data["review"].tolist():
        r = [vocab_to_int[w] for w in review.split()]
        reviews_int.append(r)

    reviews_length = [len(review) for review in reviews_int]

    padded_features = pd.DataFrame(padding_features(reviews_int, 400))
    final_data = pd.concat([padded_features, data["sentiment"]], axis = 1)

    return (final_data, len(vocab_to_int)+1)