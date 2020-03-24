import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

import itertools
import numpy as np

from sklearn import svm, linear_model
from sklearn.model_selection import train_test_split


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

        print("first")
        print(super(RankSVM, self).predict(X_trans))
        print("second")
        print(y_trans)
        return np.mean(super(RankSVM, self).predict(X_trans) == y_trans)


DATASET_SRC = "./data/loinc_dataset-v2.xlsx"

xsl = pd.ExcelFile(DATASET_SRC)
sheet_and_df = {}

NUMBER_OF_FEATURES = 43
SAMPLES_PER_DOC = 20

X = np.empty((0, NUMBER_OF_FEATURES), float)
Y = np.empty((0, 2), float)
index = 0.0

for query_name in xsl.sheet_names:
    documents = xsl.parse(query_name)['long_common_name']

    document_vectorizer = TfidfVectorizer(stop_words="english", use_idf=True)
    document_vectorizer_result = document_vectorizer.fit_transform(documents)

    document_vectorizer_df = pd.DataFrame(
        document_vectorizer_result.toarray(), columns=document_vectorizer.get_feature_names())
    document_vectorizer_array = document_vectorizer_df.to_numpy()

    # Concatenate x
    X = np.append(X, document_vectorizer_array, axis=0)

    query_vectorizer = TfidfVectorizer(stop_words="english", use_idf=True)
    query_vectorizer_result = query_vectorizer.fit_transform([query_name])

    query_vectorizer_df = pd.DataFrame(
        query_vectorizer_result.toarray(), columns=query_vectorizer.get_feature_names())
    query_vectorizer_array = query_vectorizer_df.to_numpy()

    intersection = np.intersect1d(
        document_vectorizer.get_feature_names(), query_vectorizer.get_feature_names())

    y = np.full((SAMPLES_PER_DOC), 0.0)
    # print(y)
    # print(document_vectorizer_df["blood"].to_numpy())
    for keyword in intersection:
        y = np.add(y, document_vectorizer_df[keyword].to_numpy())
        # print(y)

    # print(y)
    # print(np.c_[y, np.repeat(index, SAMPLES_PER_DOC)])
    Y = np.append(Y, np.c_[y, np.repeat(index, SAMPLES_PER_DOC)], axis=0)

    index += 1
    #     # # Used for calculate Y
    #     # np.random.seed(100)
    # query_vectorizer = TfidfVectorizer(stop_words="english", use_idf=True)
    # query_vec_result = query_vectorizer.fit_transform([query_name])

    # intersection = np.intersect1d(
    #     query_vectorizer.get_feature_names(), vectorizer.get_feature_names())

    # y = np.empty((20, 2), float)
    # for keyword in intersection:
    #     print(x_vec[keyword].to_numpy())
    #     y = np.sum(y, x_vec[keyword].to_numpy())

    # print(y)

    # print(query_vec_result.shape)
    # print(np.intersect1d(query_vectorizer.get_feature_names(),
    #                      vectorizer.get_feature_names()))

    # n_samples, n_features = 21, 43
    # true_coef = np.random.randn(n_features)
    # noise = np.random.randn(n_samples) / np.linalg.norm(true_coef)
    # y = np.dot(x_vec, true_coef)

    # # concatenate Y
    # Y = np.concatenate(
    #     (Y, np.c_[y, np.repeat(index, n_samples)]), axis=0)

    # index += 1.0

print(X)
print(X.shape)
print(Y)
print(Y.shape)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y)

print(X_train.shape)
print(Y_train)
print(X_test.shape)
print(Y_test)

rank_svm = RankSVM().fit(X_train, Y_train)

rank_svm.score(X_test, Y_test)
# print(rank_svm.coef_)
print('Performance of ranking ', rank_svm.score(X_test, Y_test))

# Calculate tf-idf
# vectorizer = TfidfVectorizer(stop_words="english", use_idf=False)
# X = vectorizer.fit_transform(documents)
# print(vectorizer.get_feature_names())
# print(pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names()).to_numpy())
