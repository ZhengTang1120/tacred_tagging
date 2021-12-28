import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from cxplain.util.test_util import TestUtil

num_words = 1024
num_samples = 500
(x_train, y_train), (x_test, y_test) = TestUtil.get_imdb(word_dictionary_size=num_words,
                                                         num_subsamples=num_samples)

from sklearn.pipeline import Pipeline
from cxplain.util.count_vectoriser import CountVectoriser
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfTransformer

explained_model = RandomForestClassifier(n_estimators=64, max_depth=5, random_state=1)

counter = CountVectoriser(num_words)
tfidf_transformer = TfidfTransformer()

explained_model = Pipeline([('counts', counter),
                            ('tfidf', tfidf_transformer),
                            ('model', explained_model)])
explained_model.fit(x_train, y_train)

print(explained_model(x_test))

from tensorflow.python.keras.losses import binary_crossentropy
from cxplain import RNNModelBuilder, WordDropMasking, CXPlain

model_builder = RNNModelBuilder(embedding_size=num_words, with_embedding=True,
                                num_layers=2, num_units=32, activation="relu", p_dropout=0.2, verbose=0,
                                batch_size=32, learning_rate=0.001, num_epochs=2, early_stopping_patience=128)
masking_operation = WordDropMasking()
loss = binary_crossentropy

from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.framework.ops import disable_eager_execution

disable_eager_execution()

explainer = CXPlain(explained_model, model_builder, masking_operation, loss)

prior_test_lengths = list(map(len, x_test))
x_train = pad_sequences(x_train, padding="post", truncating="post", dtype=int)
x_test = pad_sequences(x_test, padding="post", truncating="post", dtype=int, maxlen=x_train.shape[1])
explainer.fit(x_train, y_train);

attributions = explainer.explain(x_test)


import numpy as np
import matplotlib.pyplot as plt
from cxplain.visualisation.plot import Plot

plt.rcdefaults()

np.random.seed(909)
selected_index = np.random.randint(len(x_test))
selected_sample = x_test[selected_index]
importances = attributions[selected_index]
prior_length = prior_test_lengths[selected_index]

# Truncate to original review length prior to padding.
selected_sample = selected_sample[:prior_length]
importances = importances[:prior_length]
words = TestUtil.imdb_dictionary_indidces_to_words(selected_sample)

print(Plot.plot_attribution_nlp(words, importances))