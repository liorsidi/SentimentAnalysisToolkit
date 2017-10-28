# evaluate a models for a given pipeline and cross validation method
from itertools import cycle

from sklearn.externals import joblib
from sklearn.metrics import accuracy_score
from sklearn.metrics import auc
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_curve
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from Data import set_vec_vocabulary

import numpy as np
import pandas as pd

def evaluate_classifiers(x_raw, y_raw, labels, classifiers, cv, pipeline, print_roc, print_stats, name1, plt=None,
                         pickl_path=None):
    colors = cycle(['cyan', 'indigo', 'seagreen', 'yellow', 'blue', 'darkorange'])
    lw = 2
    # models evaluations
    clf_eval = {}
    for name, cls in classifiers.iteritems():
        clf_eval[name] = {}
        clf_eval[name]['mean_tpr'] = 0.0
        clf_eval[name]['mean_precision'] = 0.0
        clf_eval[name]['mean_accuracy'] = 0.0
        clf_eval[name]['mean_recall'] = 0.0
        mean_fpr = np.linspace(0, 1, 100)
        plt.figure(name)

    # evaluate each models on same CV configuration and pipeline
    for (train, test), color in zip(cv.split(x_raw, y_raw), colors):
        x_train = pipeline.fit_transform(x_raw[train], y_raw[train])
        x_test = pipeline.transform(x_raw[test])
        for name, cls in classifiers.iteritems():
            print("trainning " + name)
            if name == "Naive Bayes":
                x_train = x_train.toarray()
                x_test = x_test.toarray()
            probas_ = cls.fit(x_train, y_raw[train]).predict_proba(x_test)
            probas_2 = cls.fit(x_train, y_raw[train]).predict(x_test)

            # Compute ROC curve and area the curve
            fpr, tpr, thresholds = roc_curve(y_raw[test], probas_[:, 1])
            clf_eval[name]['mean_tpr'] += np.interp(mean_fpr, fpr, tpr)
            clf_eval[name]['mean_tpr'][0] = 0.0
            clf_eval[name]['roc_auc'] = auc(fpr, tpr)

            clf_eval[name]['mean_precision'] += precision_score(y_raw[test], probas_2)
            clf_eval[name]['mean_accuracy'] += accuracy_score(y_raw[test], probas_2, normalize=True)
            clf_eval[name]['mean_recall'] += recall_score(y_raw[test], probas_2, average='macro')

            plt.figure(name)
            plt.plot(fpr, tpr, lw=lw, color=color)

    for name, cls in classifiers.iteritems():
        clf_eval[name]['mean_tpr'] /= cv.n_splits
        clf_eval[name]['mean_recall'] /= cv.n_splits
        clf_eval[name]['mean_precision'] /= cv.n_splits
        clf_eval[name]['mean_accuracy'] /= cv.n_splits
        clf_eval[name]['mean_tpr'][-1] = 1.0
        clf_eval[name]['mean_auc'] = auc(mean_fpr, clf_eval[name]['mean_tpr'])
        plt.figure(name)
        plt.plot([0, 1], [0, 1], linestyle='--', lw=lw, color='k')
        plt.plot(mean_fpr, clf_eval[name]['mean_tpr'], color='g', linestyle='--',
                 label='Mean ROC (area = %0.2f)' % clf_eval[name]['mean_auc'], lw=lw)
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC for ' + name)
        plt.legend(loc="lower right")

    final_res = {}
    for name, cls in classifiers.iteritems():
        final_res[name] = {'Accuracy': clf_eval[name]['mean_accuracy'],
                           'Recall': clf_eval[name]['mean_recall'],
                           'Precision': clf_eval[name]['mean_precision'],
                           'AUC': clf_eval[name]['mean_auc']
                           }
    if print_roc:
        plt.show(name)

    eval_pipe = pd.DataFrame(final_res).transpose().sort(columns=['AUC'], axis=0, ascending=False)
    if print_stats:
        print
        print("evaluate " + name1)
        print("extractor params: " + str(pipeline.steps[0]))
        print
        print(eval_pipe)

    joblib.dump(eval_pipe, pickl_path + name + '_eval.pkl')
    return eval_pipe


def evaluate_vocabulary(x_raw, y_raw, labels, classifiers, cv, print_roc, vocab_list, prev_vec, name, replace):
    new_vec = set_vec_vocabulary(prev_vec,
                                 vocab_list,
                                 replace)
    pipe = Pipeline([('extractor', new_vec), ])
    eval_new_vec = evaluate_classifiers(x_raw, y_raw, labels, classifiers, cv, pipe, False, True, name)
    return eval_new_vec, new_vec


def get_best_param_search(x_raw, y_raw, pipeline, parameters, name, score = 'roc_auc', pickl_path=None):
    grid_search = GridSearchCV(estimator = pipeline,
                               param_grid = parameters,
                               scoring =score)

    grid_search.fit(x_raw, y_raw)
    print('best AUC: ' + str(grid_search.best_score_))
    print('best parameters:')
    print(grid_search.best_params_)
    joblib.dump(grid_search, pickl_path + name + '_param_search.pkl')
    return grid_search


# print the top words of the LDA model
def get_top_words(pipeline, n_top_words, with_selector, topic_num, optimize_pipe=None):
    feature_names = pipeline.named_steps['extractor'].get_feature_names()
    if with_selector:
        support = optimize_pipe.named_steps['selector'].get_support()
        feature_names = np.array(feature_names)[support]

    feature_list = [x for x in feature_names if x.startswith('extractor')]
    feature_list = [w.replace('extractor__', '') for w in feature_names]

    topic = pipeline.named_steps['clf'].components_[topic_num]
    return ([feature_names[i]
             for i in topic.argsort()[:-n_top_words - 1:-1]])


def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))


def print_sentiment_topics(n_topics, word1, word2, pipeline, feature_list):
    word_1 = feature_list.index(word1)
    word_2 = feature_list.index(word2)
    print(word1 + "and" + word2 + " diffrence in Topics:")
    for i in range(n_topics):
        score1 = (pipeline.named_steps['clf'].components_[i, word_1])
        score2 = (pipeline.named_steps['clf'].components_[i, word_2])
        print("topic  " + str(i) + " : " + str(abs(score1 - score2)))


def print_topics(n_topics, pipeline, with_selector, optimize_pipe=None):
    feature_names = pipeline.named_steps['extractor'].get_feature_names()
    if with_selector:
        support = optimize_pipe.named_steps['clf'].get_support()
        feature_names = np.array(feature_names)[support]

    feature_list = [x for x in feature_names if x.startswith('extractor')]
    feature_list = [w.replace('extractor__', '') for w in feature_names]

    print_sentiment_topics(n_topics, 'good', 'bad', pipeline, feature_list)
    print_top_words(pipeline.named_steps['clf'], feature_list, 10)

    def extract_calc_words_from_file(path, w2v_model, words):
        dic = {}
        with open(path) as f:
            for line in f:
                if line.startswith(';') or line == '\n':
                    continue
                line = line.split()[0]
                if line in w2v_model:
                    dic[line] = {}
                    for word in words:
                        if word in w2v_model:
                            dic[line][word + '_sim'] = w2v_model.similarity(word, line)
        return dic


def find_best_words(w2v_model,most_sim,senti_word, threshold, words_for_sim, all_words):
    return_words = {}
    for most_word, sim in most_sim:
        if most_word in all_words:
            continue
        if sim > threshold:
            return_words[most_word] = {}
            for w in words_for_sim:
                 if w in w2v_model:
                    return_words[most_word][w + '_sim'] = w2v_model.similarity(w,most_word)
    return return_words

