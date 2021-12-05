# -*- coding: utf-8 -*-

from urllib.request import urlopen;
import numpy as np;
import pandas as pd;
from sklearn.model_selection import train_test_split;
from sklearn.feature_extraction.text import CountVectorizer;
from sklearn.naive_bayes import MultinomialNB;
from sklearn.metrics import confusion_matrix;
from operator import itemgetter
from sklearn import metrics;
import sklearn.neighbors as ng; 
from sklearn.linear_model import SGDClassifier;
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE, RandomOverSampler

def frange(a,b,s):
  return [] if s > 0 and a > b or s < 0 and a < b or s==0 else [a]+frange(a+s,b,s)


def optimizeNB(X_train, y_train, X_test, y_test, maximize = "accuracy"):
    print("\nOptimizing NB")
    best_alpha = None
    best_fit_prior = None
    overall_max = 0
    for alpha in frange(3.5, 7.5, 0.01):
        for fit_prior in [True, False]:
            clf_MNB = MultinomialNB(alpha=alpha, fit_prior = fit_prior);
            clf_MNB.fit(X_train,y_train);
            if (maximize == "accuracy"): 
                curr_max = clf_MNB.score(X_test,y_test);
            elif (maximize == "precision"):
                y_test_pred = clf_MNB.predict(X_test)
                NB_metrics = metrics.classification_report(y_test, y_test_pred, output_dict = True)
                curr_max = getPrecisionSum(NB_metrics)
            if(curr_max > overall_max):
                overall_max = curr_max
                best_alpha = alpha
                best_fit_prior = fit_prior
                print("new best accuracy: " + str(curr_max))
                print("best_alpha: " + str(best_alpha) + " best_fit_prior: " + str(best_fit_prior) + "\n")
    
    return MultinomialNB(alpha=best_alpha, fit_prior=best_fit_prior)


def optimizeKNN(X_train, y_train, X_test, y_test, maximize = "accuracy"):
    print("\nOptimizing KNN")
    best_n_neighbors = None
    best_weights = None
    best_leaf_size = None
    overall_max = 0
    for n_neighbors in range(1, 16, 1):
        for weights in ["uniform", "distance"]:
            for leaf_size in range(5, 15, 1):
                clf_KNN = ng.KNeighborsClassifier(n_neighbors=n_neighbors, weights = weights,
                                                  leaf_size = leaf_size);
                clf_KNN.fit(X_train,y_train);
                if (maximize == "accuracy"): 
                    curr_max = clf_KNN.score(X_test, y_test)
                elif (maximize == "precision"):
                    y_test_pred = clf_KNN.predict(X_test)
                    KNN_metrics = metrics.classification_report(y_test, y_test_pred, output_dict = True)
                    curr_max = getPrecisionSum(KNN_metrics)
                if(curr_max > overall_max):
                    overall_max = curr_max
                    best_n_neighbors = n_neighbors
                    best_weights = weights
                    best_leaf_size = leaf_size
                    print("new best accuracy: " + str(curr_max))
                    print("best_n_neighbors: " + str(best_n_neighbors) + " best_weights: " + str(best_weights) + " best_leaf_size: " + str(best_leaf_size) + "\n")
    
    return ng.KNeighborsClassifier(n_neighbors=best_n_neighbors, weights = best_weights,
                                                  leaf_size = best_leaf_size);


def optimizeSgd(X_train, y_train, X_test, y_test, maximize = "accuracy"):
    print("\nOptimizing SGD")
    best_loss = None
    best_penalty = None
    best_alpha = None
    overall_max = 0
    for loss in ["hinge", "log", "modified_huber", "squared_hinge", "perceptron"]:
        for penalty in ["l1", "l2", "elasticnet"]:
            for alpha in frange(0.0001, 0.001, 0.0001):
                    clf_SGD = SGDClassifier(loss=loss, penalty=penalty,alpha=alpha, 
                                        random_state=2021, class_weight = "balanced", max_iter=5000);
                    clf_SGD.fit(X_train,y_train);
                    if (maximize == "accuracy"): 
                        curr_max = clf_SGD.score(X_test, y_test)
                    elif (maximize == "precision"):
                        y_test_pred = clf_SGD.predict(X_test)
                        SGD_metrics = metrics.classification_report(y_test, y_test_pred, output_dict = True)
                        curr_max = getPrecisionSum(SGD_metrics)
                    if(curr_max > overall_max):
                        overall_max = curr_max
                        best_loss = loss
                        best_penalty = penalty
                        best_alpha = alpha
                        print("new best accuracy: " + str(curr_max))
                        print("best_loss: " + str(best_loss) + " best_penalty: " + str(best_penalty) + " best_alpha: " + str(best_alpha)  + "\n")
    
    return SGDClassifier(loss=best_loss, penalty=best_penalty,alpha=best_alpha, 
                                        random_state=2021);

def plot_metrics(metrics):
    metrics.pop("accuracy")
    metrics.pop("macro avg")
    metrics.pop("weighted avg")
    sorted_pre = {k: v["precision"] for k, v in sorted(metrics.items(), key=lambda item: item[1]["precision"])}
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    langs = list(sorted_pre.keys())
    students = list(sorted_pre.values())
    ax.bar(langs,students)
    plt.xticks(rotation=90)
    plt.show()
    
def getPrecisionSum(metrics):
    metrics.pop("accuracy")
    metrics.pop("macro avg")
    metrics.pop("weighted avg")
    return sum({x["precision"] for x in metrics.values()})
    
def show_most_informative_features(vectorizer, clf, n=5):
    feature_names = vectorizer.get_feature_names()
    for i in range(0, len(clf.classes_)):   
        coefs_with_fns = sorted(zip(clf.coef_[i], feature_names))
        top = coefs_with_fns[:-(n + 1):-1]
        print("Most important words for " + clf.classes_[i])
        for (coef, word) in top:
            print ("\tword: " + word + "\t\t\tcoef: " + str(coef))
    
            

#KONSTANSOK, ezek a paraméterek szabadon állíthatók

#Előfeldolgozás konstansai
MIN_WORD_COUNT = 3                   #Minimum milyen hosszú legyen a szöveg
NUM_OF_CHARACTERS = 10               #Hány karaktert vizsgálunk


#NB konstansai
NB_DO_OPTIMIZE = False
NB_ALPHA = 5.05;
NB_FIT_PRIOR = True


#KNN konstansai
KNN_DO_OPTIMIZE = False
KNN_NUMBER_OF_NEIGHBOURS = 9
KNN_WEIGHTS = "distance"
KNN_LEAF_SIZE = 10


#SGD konstansai
SGD_DO_OPTIMIZE = True
SGD_LOSS = "modified_huber"
SGD_PENALTY = "l2"
SGD_ALPHA = 0.001


#Vektorizálás konstansai
MIN_DF = 0.0001
MAX_DF = 0.8


#Dataset betöltése
url = 'https://raw.githubusercontent.com/Rombeii/Szoveg-es-webbanyaszat/main/simpsons_dataset.csv';
raw_data = urlopen(url);
df=pd.read_csv(raw_data, sep=',', error_bad_lines=False, header = 0, skip_blank_lines=True).dropna().to_numpy()


#Előfeldolgozás
unique, counts = np.unique(df[:,0], return_counts=True)
dict_with_count = dict(zip(unique, counts))
sorted_elements = sorted(dict_with_count.items(), key=itemgetter(1), reverse=True)
#Top NUM_OF_CHARACTERS kiválasztása
top = np.array({x[0] for x in sorted_elements[:NUM_OF_CHARACTERS]})
df = np.array([x for x in df if x[0] in top.tolist()])
#Felesleges whitespacek törlése a szöveg elejéről és végéről
df = np.char.strip(df.astype(str))
#Rövidebb, mint MIN_WORD_COUNT szövegek törlése
df = np.array([x for x in df if len(x[1].split()) > MIN_WORD_COUNT])


#Az előfeldolgozott datasetből kiszedjük a karaktereket és a hozzájuk tartozó sorokat
characters = df[:,0]
lines = df[:,1]


#Vektorizálás
vectorizer = CountVectorizer(stop_words='english', max_df=MAX_DF, min_df=MIN_DF);
DT_train = vectorizer.fit_transform(lines)
vocabulary_dict = vectorizer.vocabulary_
vocabulary_list = vectorizer.get_feature_names()
n_words = DT_train.shape[1]


#Dataset szétválasztása test-re és train-re
X_train, X_test, y_train, y_test = train_test_split(DT_train,characters, test_size=0.3,
                                shuffle = True, random_state=2021)

os =  SMOTE(sampling_strategy = "minority", random_state = 2021)
X_train, y_train = os.fit_resample(X_train, y_train)

#NB illesztése
clf_MNB = optimizeNB(X_train, y_train, X_test, y_test, "accuracy") if NB_DO_OPTIMIZE else MultinomialNB(alpha=NB_ALPHA, fit_prior =  NB_FIT_PRIOR);
clf_MNB.fit(X_train,y_train)
NB_train_accuracy = clf_MNB.score(X_train,y_train)
NB_test_accuracy = clf_MNB.score(X_test,y_test)

y_train_pred = clf_MNB.predict(X_train)
NB_cm_train = confusion_matrix(y_train, y_train_pred)
y_test_pred = clf_MNB.predict(X_test)
NB_cm_test = confusion_matrix(y_test, y_test_pred)
print(metrics.classification_report(y_test, y_test_pred))
NB_metrics = metrics.classification_report(y_test, y_test_pred, output_dict = True)
plot_metrics(NB_metrics)
show_most_informative_features(vectorizer, clf_MNB)


#KNN illesztése
#clf_KNN = optimizeKNN(X_train, y_train, X_test, y_test, "accuracy") if KNN_DO_OPTIMIZE else ng.KNeighborsClassifier(n_neighbors=KNN_NUMBER_OF_NEIGHBOURS,
#                                                                                                        weights = KNN_WEIGHTS, leaf_size = KNN_LEAF_SIZE);
#clf_KNN.fit(X_train,y_train);
#ds_train_pred = clf_KNN.predict(X_train)
#KNN_train_accuracy = clf_KNN.score(X_train, y_train)
#KNN_test_accuracy = clf_KNN.score(X_test, y_test)

#y_train_pred = clf_KNN.predict(X_train)
#KNN_cm_train = confusion_matrix(y_train, y_train_pred)
#y_test_pred = clf_KNN.predict(X_test)
#KNN_cm_test = confusion_matrix(y_test, y_test_pred)
#print(metrics.classification_report(y_test, y_test_pred))
#KNN_metrics = metrics.classification_report(y_test, y_test_pred, output_dict = True)
#plot_metrics(KNN_metrics)


#SGD illesztése
clf_SGD = optimizeSgd(X_train, y_train, X_test, y_test, "accuracy") if SGD_DO_OPTIMIZE else SGDClassifier(loss=SGD_LOSS, penalty=SGD_PENALTY, alpha=SGD_ALPHA, 
                                                                                              random_state=2021, max_iter = 5000);
clf_SGD.fit(X_train,y_train);   
ds_train_pred = clf_SGD.predict(X_train);
SGD_train_accuracy = clf_SGD.score(X_train, y_train)
SGD_test_accuracy = clf_SGD.score(X_test, y_test)

y_train_pred = clf_SGD.predict(X_train)
SGD_cm_train = confusion_matrix(y_train, y_train_pred)
y_test_pred = clf_SGD.predict(X_test)
SGD_cm_test = confusion_matrix(y_test, y_test_pred)
print(metrics.classification_report(y_test, y_test_pred))
SGD_metrics = metrics.classification_report(y_test, y_test_pred, output_dict = True)
plot_metrics(SGD_metrics)