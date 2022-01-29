from __future__ import division
from codecs import open
from sklearn import tree
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import recall_score, f1_score, precision_score, accuracy_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import plot_confusion_matrix


# returns a string of the scores and confusion matrix
def to_string(true_labels, predicted_labels):
    return "\nConfusion matrix:\n" + np.array2string(
        confusion_matrix(true_labels, predicted_labels)) + "\nPrecision score: " + str(
        precision_score(true_labels, predicted_labels, pos_label="pos")) + "\nRecall score: " + str(
        recall_score(true_labels, predicted_labels, pos_label="pos")) + "\nF1 score: " + str(
        f1_score(true_labels, predicted_labels, pos_label="pos")) + "\nAccuracy score: " + str(
        accuracy_score(true_labels, predicted_labels))


# read documents and remove document identifier and topic label
def read_documents(doc_file):
    doc = []
    lab = []
    with open(doc_file, encoding='utf-8') as f:
        for line in f:
            words = line.strip().split()
            doc.append(words[3:])
            lab.append(words[1])
        doc = [" ".join(l) for l in doc]
    return doc, lab


# plot the number of occurrences of neg and pos in a samples
def plt_pos_neg(labl):
    num_pos = 0
    num_neg = 0
    for label in labl:
        if label == "pos":
            num_pos = num_pos + 1
        else:
            num_neg = num_neg + 1

    x = ["pos", "neg"]
    y = [num_pos, num_neg]
    plt.barh(x, y)

    for index, value in enumerate(y):
        plt.text(value, index, str(value))


# get words and labels
docs, labels = read_documents("data.txt")

# count occurrences of pos and neg the entire dataset
plt_pos_neg(labels)
plt.title("Pos and Neg ratio in entire dataset")
plt.show()

# indices to keep track of original position
indices = range(len(docs))

# split data into training(80%) and test data(20%)
training_docs, test_docs, training_labels, test_labels, indices_train, indices_test = train_test_split(docs, labels,
                                                                                                       indices,
                                                                                                       test_size=0.2)
# count occurrences of pos and neg in training set
plt_pos_neg(training_labels)
plt.title("Pos and Neg ratio in training dataset")
plt.show()

# count occurrences of pos and neg in test set
plt_pos_neg(test_labels)
plt.title("Pos and Neg ratio in test dataset")
plt.show()

# countvectorized and then tfidf the data
vectorizer = TfidfVectorizer()
training_docs_tfidf = vectorizer.fit_transform(training_docs)
test_docs_tfidf = vectorizer.transform(test_docs)

# fit the data in naive bayes classifier
nb = MultinomialNB()
clf_naive = nb.fit(training_docs_tfidf, training_labels)
# predict using naive bayes
predicted_naive = clf_naive.predict(test_docs_tfidf)

# plot confusion matrix of NB
plot_confusion_matrix(clf_naive, test_docs_tfidf, test_labels)
plt.title('Naives Bayes')
plt.show()

# write naive bayes results and performance to file
file_1 = open("NaiveBayes-Data", "w")
for x in range(len(indices_test)):
    file_1.write(str(indices_test[x]) + ", " + predicted_naive[x] + "\n")
file_1.write(to_string(test_labels, predicted_naive))
file_1.close()

# base dt fit and prediction
clf_base_dt = DecisionTreeClassifier(criterion="entropy").fit(training_docs_tfidf, training_labels)
predicted_base_dt = clf_base_dt.predict(test_docs_tfidf)

# plot confusion matrix of base dt
plot_confusion_matrix(clf_base_dt, test_docs_tfidf, test_labels)
plt.title('Base DT')
plt.show()

# tree plot of base dt
tree.plot_tree(clf_base_dt)
plt.title('Base DT')
plt.show()

# write base dt performance and results to file
file_2 = open("BaseDT-Data", "w")
for x in range(len(indices_test)):
    file_2.write(str(indices_test[x]) + ", " + predicted_base_dt[x] + "\n")
file_2.write(to_string(test_labels, predicted_base_dt))
file_2.close()

# best dt classification and prediction
clf_best_dt = DecisionTreeClassifier(criterion="gini", splitter="random", max_depth=30).fit(training_docs_tfidf,
                                                                                            training_labels)
predicted_best_dt = clf_best_dt.predict(test_docs_tfidf)

# plot confusion matrix of best dt
plot_confusion_matrix(clf_best_dt, test_docs_tfidf, test_labels)
plt.title('Best DT')
plt.show()

# tree plot of best dt
tree.plot_tree(clf_best_dt)
plt.title('Best DT')
plt.show()

# write best dt performance and result to file
file_3 = open("BestDT-Data", "w")
for x in range(len(indices_test)):
    file_3.write(str(indices_test[x]) + ", " + predicted_best_dt[x] + "\n")
file_3.write(to_string(test_labels, predicted_best_dt))
file_3.close()
