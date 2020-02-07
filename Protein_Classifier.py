from sklearn import svm
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn import preprocessing
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.feature_extraction.text import CountVectorizer


def classify(multiClassifier, test_X, test_Y):
    # test_X = scaler.transform(test_X)

    output = multiClassifier.predict(test_X)
    tp = 0.0
    fp = 0.0

    for i in range(len(output)):
        if (output[i] == test_Y[i]):
            tp += 1
        else:
            fp += 1

    # print("TP: " + str(tp))
    # print("FP: " + str(fp))

    additive_accuracy = (tp) / (tp + fp + 0.00001)

    print(" accuracy " + str(additive_accuracy * 100.0) + " percent\n")
    print('\n')


def get_classifier(training_data, i, j):
    labelEncoder = preprocessing.LabelEncoder()
    labelEncoder.fit(training_data.family)
    training_data.family = labelEncoder.transform(training_data.family)

    labelEncoder.fit(training_data.descriptor)
    training_data.descriptor = labelEncoder.transform(training_data.descriptor)

    labelEncoder.fit(training_data.classification)
    training_data.classification = labelEncoder.transform(training_data.classification)

    labelEncoder.fit(training_data.classification)
    training_data.classification = labelEncoder.transform(training_data.classification)

    labelEncoder.fit(training_data.name)
    training_data.name = labelEncoder.transform(training_data.name)

    oneHotEncoder = preprocessing.OneHotEncoder(sparse=False)
    integer_encoded_seq = labelEncoder.fit_transform(training_data.sequence)
    integer_encoded_seq = integer_encoded_seq.reshape(len(integer_encoded_seq), 1)
    training_data.sequence = oneHotEncoder.fit_transform(integer_encoded_seq)

    X = training_data.loc[:, ['cellAngleAlpha', 'sequence', 'descriptor', 'name',
                              'cellAngleBeta', 'cellAngleGamma', 'cellLengthA', 'cellLengthB', 'cellLengthC',
                              'resolution', 'rValueWork', 'rValueFree', 'classification', 'atomSiteCount',
                              'molecularWeight', 'numberOfPolypeptideChains', 'residueCount', 'taxonomyID']]

    if i == 1:
        Y = training_data.classID
        print("\nClass:")
        X = training_data.loc[:, ['cellAngleAlpha', 'sequence', 'descriptor', 'name',
                                  'cellAngleBeta', 'cellAngleGamma', 'cellLengthA', 'cellLengthB', 'cellLengthC',
                                  'resolution', 'rValueWork', 'rValueFree', 'classification', 'atomSiteCount',
                                  'molecularWeight', 'numberOfPolypeptideChains', 'residueCount', 'family',
                                  'superFamily', 'fold']]
    elif i == 2:
        Y = training_data.family
        print("\nFamily:")
    elif i == 3:
        Y = training_data.superFamily
        print("\nSuperFamily:")
        X = training_data.loc[:, ['cellAngleAlpha', 'sequence',
                                  'cellAngleBeta', 'cellAngleGamma', 'cellLengthA', 'cellLengthB', 'cellLengthC',
                                  'resolution', 'rValueWork', 'rValueFree', 'classification', 'atomSiteCount',
                                  'molecularWeight', 'numberOfPolypeptideChains', 'residueCount', 'family']]
    else:
        print("\nFold:")
        Y = training_data.fold
        X = training_data.loc[:, ['cellAngleAlpha', 'sequence',
                                  'cellAngleBeta', 'cellAngleGamma', 'cellLengthA', 'cellLengthB', 'cellLengthC',
                                  'resolution', 'rValueWork', 'rValueFree', 'classification', 'atomSiteCount',
                                  'molecularWeight', 'numberOfPolypeptideChains', 'residueCount', 'family',
                                  'superFamily']]

    X = preprocessing.StandardScaler().fit_transform(X)

    # print(training_data.sequence)
    # scaler = preprocessing.StandardScaler().fit(X)
    # X = scaler.transform(X)
    # pca = IncrementalPCA(n_components=5, batch_size=1000)

    clf = tree.DecisionTreeClassifier()
    if j == 1:
        clf = tree.DecisionTreeClassifier()
        print("Decision Tree Classifier: ")
    elif j == 2:
        clf = BaggingClassifier(KNeighborsClassifier(n_neighbors=1), max_samples=.8, max_features=.8)
        print("Bagging with  KNN Classifier: ")
    elif j == 3:
        clf = KNeighborsClassifier(n_neighbors=1)
        print("KNN Classifier: k = 3")
    elif j == 4:
        clf = RandomForestClassifier(max_depth=3, n_estimators=30, max_features=.5)
        print("Random Forest Classifier: depth = 5, est = 20 and max_features = 0.8")
    elif j == 5:
        clf = svm.SVC()
        print("SVM Classifier:")
    elif j == 6:
        clf = MLPClassifier(hidden_layer_sizes=(7, 6, 5, 4, 3, 1))
        print("MLP Classifier: hidden layer = 6")
    elif j == 7:
        clf = GaussianNB()
        print("Gaussian NB classifier: ")
    elif j == 8:
        print("Logistic regression classifier")
        clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial')
    elif j == 9:
        print("ADA boosting classifier: ")
        clf = AdaBoostClassifier(base_estimator=None, n_estimators=50, learning_rate=1.0, random_state=None)
    elif j == 10:
        clf = BernoulliNB()
        print("Bernouli Naive Bayes  classifier: ")
    elif j == 11:
        print("Extra Tree Classifier")
        clf = ExtraTreesClassifier(n_estimators=20, max_depth=None,
                                   min_samples_split=5, random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.15, random_state=42)

    clf.fit(X_train, y_train)
    # classify_train_test_split(clf, X_test, y_test )
    return clf, X_test, y_test


training_data = pd.read_csv('scopOutput.txt', delimiter=',', index_col='sunID')

j = 1
while j < 12:
    i = 1
    while i < 5:
        classifier, test_X, test_Y = get_classifier(training_data, i, j)
        classify(classifier, test_X, test_Y)
        # print(i)
        i = i + 1
    # print(j)
    j = j + 1

# sunID,SID ,SCCS_ID,family,superFamily,fold,classID,species,name,descriptor,sequence,cellAngleAlpha,cellAngleBeta,cellAngleGamma,cellLengthA,cellLengthB,cellLengthC,resolution,rValueWork,rValueFree,classification,atomSiteCount,molecularWeight,numberOfPolypeptideChains,residueCount,taxonomyID
