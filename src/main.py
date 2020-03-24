import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

import itertools
import numpy as np

from sklearn import svm, linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity


def transform_pairwise(X, y):
    """Transforms data into pairs with balanced labels for ranking
    Transforms a n-class ranking problem into a two-class classification
    problem. Subclasses implementing particular strategies for choosing
    pairs should override this method.
    In this method, all pairs are choosen, except for those that have the
    same target value. The output is an array of balanced classes, i.e.
    there are the same number of -1 as +1
    Parameters
    ----------
    X : array, shape (n_samples, n_features)
        The data
    y : array, shape (n_samples,) or (n_samples, 2)
        Target labels. If it's a 2D array, the second column represents
        the grouping of samples, i.e., samples with different groups will
        not be considered.
    Returns
    -------
    X_trans : array, shape (k, n_feaures)
        Data as pairs
    y_trans : array, shape (k,)
        Output class labels, where classes have values {-1, +1}
    """
    X_new = []
    y_new = []
    y = np.asarray(y)
    if y.ndim == 1:
        y = np.c_[y, np.ones(y.shape[0])]
    comb = itertools.combinations(range(X.shape[0]), 2)
    for k, (i, j) in enumerate(comb):
        if y[i, 0] == y[j, 0] or y[i, 1] != y[j, 1]:
            # skip if same target or different group
            continue
        X_new.append(X[i] - X[j])
        y_new.append(np.sign(y[i, 0] - y[j, 0]))
        # output balanced classes
        if y_new[-1] != (-1) ** k:
            y_new[-1] = - y_new[-1]
            X_new[-1] = - X_new[-1]
    return np.asarray(X_new), np.asarray(y_new).ravel()


class RankSVM(svm.LinearSVC):
    """Performs pairwise ranking with an underlying LinearSVC model
    Input should be a n-class ranking problem, this object will convert it
    into a two-class classification problem, a setting known as
    `pairwise ranking`.
    See object :ref:`svm.LinearSVC` for a full description of parameters.
    """

    def fit(self, X, y):
        """
        Fit a pairwise ranking model.
        Parameters
        ----------
        X : array, shape (n_samples, n_features)
        y : array, shape (n_samples,) or (n_samples, 2)
        Returns
        -------
        self
        """
        X_trans, y_trans = transform_pairwise(X, y)
        super(RankSVM, self).fit(X_trans, y_trans)
        return self

    def predict(self, X):
        """
        Predict an ordering on X. For a list of n samples, this method
        returns a list from 0 to n-1 with the relative order of the rows of X.
        Parameters
        ----------
        X : array, shape (n_samples, n_features)
        Returns
        -------
        ord : array, shape (n_samples,)
            Returns a list of integers representing the relative order of
            the rows in X.
        """
        if hasattr(self, 'coef_'):
            return np.argsort(np.dot(X, self.coef_.T).flatten())
        else:
            raise ValueError("Must call fit() prior to predict()")

    def score(self, X, y):
        """
        Because we transformed into a pairwise problem, chance level is at 0.5
        """
        X_trans, y_trans = transform_pairwise(X, y)
        return np.mean(super(RankSVM, self).predict(X_trans) == y_trans)


DATASET_SRC = "./data/loinc_dataset-v2.xlsx"

xsl = pd.ExcelFile(DATASET_SRC)
sheet_and_df = {}

NUMBER_OF_FEATURES = 43
SAMPLES_PER_DOC = 20


X = np.empty((0, NUMBER_OF_FEATURES), float)
Y = np.empty((0, 2), float)
index = 0.0

full_docs = []

for query_name in xsl.sheet_names:
    documents = xsl.parse(query_name)['long_common_name']
    full_docs = documents

    document_vectorizer = TfidfVectorizer(stop_words="english", use_idf=True)
    document_vectorizer_result = document_vectorizer.fit_transform(documents)

    document_vectorizer_df = pd.DataFrame(
        document_vectorizer_result.toarray(), columns=document_vectorizer.get_feature_names())

    query_vectorizer = TfidfVectorizer(stop_words="english", use_idf=True)
    query_vectorizer_result = query_vectorizer.fit_transform([query_name])

    query_vectorizer_df = pd.DataFrame(
        query_vectorizer_result.toarray(), columns=query_vectorizer.get_feature_names())
    query_vectorizer_array = query_vectorizer_df.to_numpy()

    document_vectorizer_array = document_vectorizer_df.to_numpy()

    intersection = np.intersect1d(
        document_vectorizer.get_feature_names(), query_vectorizer.get_feature_names())

    # Concatenate y
    y = np.full((SAMPLES_PER_DOC), 0.0)
    for keyword in intersection:
        y = np.add(y, document_vectorizer_df[keyword].to_numpy())

    Y = np.append(Y, np.c_[y, np.repeat(index, SAMPLES_PER_DOC)], axis=0)
    # Concatenate x
    X = np.append(X, document_vectorizer_array, axis=0)

    index += 1

rank_svm = RankSVM().fit(X, Y)

search_query = ""
while search_query != "exit":

    search_query = input("Enter the search keywords: (enter exit for quit) ")
    print()

    if (search_query == "exit"):
        break

    X = np.empty((0, NUMBER_OF_FEATURES), float)
    Y = np.empty((0, 2), float)

    document_vectorizer = TfidfVectorizer(stop_words="english", use_idf=True)
    document_vectorizer_result = document_vectorizer.fit_transform(full_docs)

    document_vectorizer_df = pd.DataFrame(
        document_vectorizer_result.toarray(), columns=document_vectorizer.get_feature_names())

    query_vectorizer = TfidfVectorizer(stop_words="english", use_idf=True)
    query_vectorizer_result = query_vectorizer.fit_transform([search_query])

    query_vectorizer_df = pd.DataFrame(
        query_vectorizer_result.toarray(), columns=query_vectorizer.get_feature_names())
    query_vectorizer_array = query_vectorizer_df.to_numpy()

    # Concatenate x
    X = np.append(X, document_vectorizer_array, axis=0)

    intersection = np.intersect1d(
        document_vectorizer.get_feature_names(), query_vectorizer.get_feature_names())

    # Concatenate y
    y = np.full((SAMPLES_PER_DOC), 0.0)
    for keyword in intersection:
        y = np.add(y, document_vectorizer_df[keyword].to_numpy())

    Y = np.append(Y, np.c_[y, np.repeat(index, SAMPLES_PER_DOC)], axis=0)

    results = rank_svm.predict(X)

    for result in results:
        print(full_docs[result])
