from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
import sklearn_crfsuite
from models.MLP import model_MLP
from models.LSTM import LSTM
from models.DNN import DNN
from models.GRU import GRU
from models.BiLSTM import BiLSTM
from models.LSTM_Attention import AttentionLSTM
from models.RNN import RNN
from models.TextCNN import TextCNN
from models.TextRCNN import TextRCNN
from models.VDCNN import VDCNN
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from models.MachineDeepLearningTrain import deepLearningTrain
from models.MachineDeepLearningTrain import machineLearningTrain


def train_by_model(train_data, train_y, req):

    test_data = pd.read_csv("feature_extraction/after_feature_data/test_data.csv", header=None).to_numpy()
    test_y = pd.read_csv("feature_extraction/after_feature_data/test_labels.csv", header=None).to_numpy()
    test_y = test_y.ravel()

    selected_model = req.form.get('modelCheckbox')
    feature_num = train_data.shape[1]
    class_num = len(np.unique(train_y))
    acc = 0
    if selected_model == "SVM":
        print("This is SVM")
        model = SVC(probability=True)
        acc = machineLearningTrain(model,train_data, train_y, test_data, test_y)
    if selected_model == "RandomForest":
        print("This is Random Forest")
        model = RandomForestClassifier()
        acc = machineLearningTrain(model, train_data, train_y, test_data, test_y)
    if selected_model == "DecisionTrees":
        print("This is Decision Trees")
        model = DecisionTreeClassifier()
        acc = machineLearningTrain(model, train_data, train_y, test_data, test_y)
    if selected_model == "NaiveBayes":
        print("This is Naive Bayes")
        model = GaussianNB()
        acc = machineLearningTrain(model, train_data, train_y, test_data, test_y)
    if selected_model == "KNN":
        print("This is KNN")
        model = KNeighborsClassifier()
        acc = machineLearningTrain(model, train_data, train_y, test_data, test_y)
    if selected_model == "MLP":
        print("This is MLP")
        model = model_MLP(feature_num, class_num)
        acc = deepLearningTrain(model, train_data, train_y, test_data, test_y)
    if selected_model == "CRF":
        print("This is CRF")
        model = sklearn_crfsuite.CRF(
            algorithm='lbfgs',
            c1=0.1,
            c2=0.1,
            max_iterations=100,
            all_possible_transitions=False
        )
        acc = machineLearningTrain(model, train_data, train_y, test_data, test_y, isCRF=True)
    if selected_model == "GradientBoostingTrees":
        print("This is Gradient Boosting Trees")
        model = GradientBoostingClassifier()
        acc = machineLearningTrain(model, train_data, train_y, test_data, test_y)
    ########################################

    if selected_model == "DNN":
        print("This is DNN")
        model = DNN(feature_num, class_num)
        acc = deepLearningTrain(model, train_data, train_y, test_data, test_y)
    if selected_model == "RNN":
        print("This is RNN")
        model = RNN(feature_num, class_num)
        acc = deepLearningTrain(model, train_data, train_y, test_data, test_y)
    if selected_model == "LSTM":
        print("This is LSTM")
        model = LSTM(feature_num, class_num)
        acc = deepLearningTrain(model, train_data, train_y, test_data, test_y)
    if selected_model == "BiLSTM":
        print("This is BiLSTM")
        model = BiLSTM(feature_num, class_num)
        acc = deepLearningTrain(model, train_data, train_y, test_data, test_y)
    if selected_model == "LSTM-Attention":
        print("This is LSTM-Attention")
        model = AttentionLSTM(feature_num, class_num)
        acc = deepLearningTrain(model, train_data, train_y, test_data, test_y)
    if selected_model == "GRU":
        print("This is GRU")
        model = GRU(feature_num, class_num)
        acc = deepLearningTrain(model, train_data, train_y, test_data, test_y)
    if selected_model == "TextCNN":
        print("This is TextCNN")
        model = TextCNN(feature_num, class_num)
        acc = deepLearningTrain(model, train_data, train_y, test_data, test_y)
    if selected_model == "TextRCNN":
        print("This is TextRCNN")
        model = TextRCNN(feature_num, class_num)
        acc = deepLearningTrain(model, train_data, train_y, test_data, test_y)
    if selected_model == "VDCNN":
        print("This is VDCNN")
        model = VDCNN(100, class_num)
        acc = deepLearningTrain(model, train_data, train_y, test_data, test_y)
    if selected_model == "None":
        print("None")
        acc = 0
    print(f"acc:{acc}")

    return acc, selected_model
