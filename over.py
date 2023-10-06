import pandas as pd
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression
from imblearn.pipeline import make_pipeline as make_pipeline_with_sampler
from sklearn.cluster import KMeans
# KMeans.DEFAULT_N_INIT = 3  # 设置全局的默认n_init值
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_transformer
from sklearn.compose import make_column_selector as selector
from itertools import product
import dataset
import torch
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import VotingClassifier,RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier,BaggingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
import math
# print(y_res.value_counts())
index = []
scoring = ["accuracy", "balanced_accuracy"]
scores = {"Accuracy": [], "Balanced accuracy": []}
RANDOMSTATE=42


from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
from randomover import RandomOverSampler
from smoteover import SMOTEOverSampler
from adasynover import ADASYNOverSampler

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

# 定义1D CNN模型
class CNN1DClassifier(nn.Module):
    def __init__(self, input_length, num_classes):
        super(CNN1DClassifier, self).__init__()
        # First convolution and pooling
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)

        # Second convolution and pooling
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)

        # Third convolution and pooling
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)

        # Fourth convolution and pooling
        self.conv4 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.pool4 = nn.MaxPool1d(kernel_size=1, stride=2, padding=0)

        # Fully connected layers
        self.fc1 = nn.Linear(128 , 256)  # Adjusted the size due to extra pooling layers
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.pool4(F.relu(self.conv4(x)))
        x = x.view(-1, 128 * (x.shape[2]))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, d_model=128, nhead=8, num_encoder_layers=2):
        super(TransformerClassifier, self).__init__()

        # Initial linear layer to transform input to d_model
        self.init_linear = nn.Linear(input_dim, d_model)

        # Transformer encoder
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,batch_first=True),
            num_layers=num_encoder_layers
        )

        # Classifier head
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = x.permute( 0,2,1)
        x = self.init_linear(x)
        x = self.transformer(x)
        x = x.mean(dim=1)  # Average along the sequence length dimension
        x = self.classifier(x)
        return x

testrate=0.4

from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.naive_bayes import GaussianNB
def train_and_evaluate(X, Y, test_rate, random_state):
    """
    Train and evaluate the Logistic Regression model, and return accuracy and balanced accuracy.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_rate, random_state=random_state)

    # clf= LogisticRegression(random_state=random_state)
    clf = GaussianNB()
    # clf= DecisionTreeClassifier(random_state=random_state)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    accuracy = 1 - sum(abs(y_pred - y_test)) / len(y_pred)
    bal_acc = balanced_accuracy_score(y_pred, y_test)



    # # Convert data to PyTorch tensors
    # if isinstance(X_train, pd.DataFrame):
    #     X_train = X_train.values
    # if isinstance(X_test, pd.DataFrame):
    #     X_test = X_test.values
    #
    # X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
    # X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
    #
    # # 对 y_train 和 y_test 进行检查和转换
    # if isinstance(y_train, (pd.DataFrame, pd.Series)):
    #     y_train = y_train.values
    # if isinstance(y_test, (pd.DataFrame, pd.Series)):
    #     y_test = y_test.values
    #
    # y_train = torch.tensor(y_train, dtype=torch.long)
    # y_test = torch.tensor(y_test, dtype=torch.long)
    #
    # # Create DataLoader
    # train_data = TensorDataset(X_train, y_train)
    # train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    #
    # # Initialize the CNN model
    # input_length = X_train.shape[-1]
    # num_classes = 2
    #
    # # model = CNN1DClassifier(input_length, num_classes)
    # model = TransformerClassifier(input_dim=1, num_classes=2)
    #
    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=0.003)
    #
    # # Training the model
    # best_accuracy = 0.0  # Initialize best accuracy with 0
    # best_model_state = None  # This will store the best model's state
    #
    # # Training the model
    # for epoch in range(2):  # Assuming 10 epochs, you can adjust this
    #     # Training Phase
    #     model.train()  # Set the model to training mode
    #     for inputs, labels in train_loader:
    #         optimizer.zero_grad()
    #         outputs = model(inputs)
    #         loss = criterion(outputs, labels)
    #         loss.backward()
    #         optimizer.step()
    #
    #     # Evaluation Phase
    #     model.eval()  # Set the model to evaluation mode
    #     with torch.no_grad():
    #         outputs = model(X_test)
    #         _, y_pred = torch.max(outputs, 1)
    #         accuracy = (y_pred == y_test).float().mean().item()
    #
    #     print(f"Epoch {epoch + 1}/{2} - Validation Accuracy: {accuracy:.4f}")
    #
    #     # Check if this epoch has the best accuracy
    #     if accuracy > best_accuracy:
    #         best_accuracy = accuracy
    #         best_model_state = model.state_dict()
    #
    # # Load the best model state
    # model.load_state_dict(best_model_state)
    #
    # # Final evaluation with the best model
    # with torch.no_grad():
    #     outputs = model(X_test)
    #     _, y_pred = torch.max(outputs, 1)
    #     accuracy = (y_pred == y_test).float().mean().item()
    #     bal_acc = balanced_accuracy_score(y_test.numpy(), y_pred.numpy())

    return accuracy, bal_acc


import matplotlib.pyplot as plt
import numpy as np
# Lists to store results
acc1_list, bal_acc1_list = [], []
acc2_list, bal_acc2_list = [], []
acc3_list, bal_acc3_list = [], []
acc4_list, bal_acc4_list = [], []
acc5_list, bal_acc5_list = [], []
acc6_list, bal_acc6_list = [], []
acc7_list, bal_acc7_list = [], []
ind = np.linspace(0, 1, 12)
# y_samples = np.random.uniform(0, 30, 30)
# ind = y_samples / (1 + y_samples)

accident_weight=[]
for i in range(len(ind)-2):
    # print([0, ind[i+1], 1-ind[i+1]])
    X, Y, W = dataset.datasetprepare(safety_weight=[0, ind[i+1], 1-ind[i+1]])
    accident_weight.append(ind[i+1])

    # Without resampling
    acc1, bal_acc1 = train_and_evaluate(X, Y, testrate, RANDOMSTATE)
    acc1_list.append(acc1)
    bal_acc1_list.append(bal_acc1)




    # With SMOTEOverSampler (with weights)
    ros = SMOTEOverSampler(random_state=RANDOMSTATE)
    X_resampled, y_resampled = ros.fit_resample(X, Y,W=W)
    acc2, bal_acc2 = train_and_evaluate(X_resampled, y_resampled, testrate, RANDOMSTATE)
    acc2_list.append(acc2)
    bal_acc2_list.append(bal_acc2)

    # With SMOTEOverSampler (no weights)
    ros = SMOTEOverSampler(random_state=RANDOMSTATE)
    X_resampled, y_resampled = ros.fit_resample(X, Y,W=None)
    acc3, bal_acc3 = train_and_evaluate(X_resampled, y_resampled, testrate, RANDOMSTATE)
    acc3_list.append(acc3)
    bal_acc3_list.append(bal_acc3)


    # With RandomOverSampler (with weights)
    ros = RandomOverSampler(random_state=RANDOMSTATE)
    X_resampled_w, y_resampled_w = ros.fit_resample(X, Y, W=W)
    acc4, bal_acc4 = train_and_evaluate(X_resampled_w, y_resampled_w, testrate, RANDOMSTATE)
    acc4_list.append(acc4)
    bal_acc4_list.append(bal_acc4)

    # With RandomOverSampler (no weights)
    ros = RandomOverSampler(random_state=RANDOMSTATE)
    X_resampled_w, y_resampled_w = ros.fit_resample(X, Y, W=None)
    acc5, bal_acc5 = train_and_evaluate(X_resampled_w, y_resampled_w, testrate, RANDOMSTATE)
    acc5_list.append(acc5)
    bal_acc5_list.append(bal_acc5)



    # With ADASYNOverSampler (with weights)
    ros = ADASYNOverSampler(random_state=RANDOMSTATE)
    X_resampled_w, y_resampled_w = ros.fit_resample(X, Y, W=W)
    acc6, bal_acc6= train_and_evaluate(X_resampled_w, y_resampled_w, testrate, RANDOMSTATE)
    acc6_list.append(acc6)
    bal_acc6_list.append(bal_acc6)

    # With ADASYNOverSampler (no weights)
    ros = ADASYNOverSampler(random_state=RANDOMSTATE)
    X_resampled_w, y_resampled_w = ros.fit_resample(X, Y, W=None)
    acc7, bal_acc7 = train_and_evaluate(X_resampled_w, y_resampled_w, testrate, RANDOMSTATE)
    acc7_list.append(acc7)
    bal_acc7_list.append(bal_acc7)

# Plotting
plt.figure(figsize=(8, 6))

# plt.subplot(1, 2, 1)
# plt.plot(ind, acc1_list, label='No Resampling', marker='o')
# plt.plot(ind, acc2_list, label='OverSampler No Weights', marker='x')
# plt.plot(ind, acc3_list, label='OverSampler With Weights', marker='.')
# plt.title('Accuracy vs safety_weight')
# plt.xlabel('safety_weight')
# plt.ylabel('Accuracy')
# plt.legend()

# plt.subplot(1, 2, 2)
ind=ind[1:-1]
accident_weight=np.array(accident_weight)
ind=accident_weight


average = sum(bal_acc1_list) / len(bal_acc1_list)
bal_acc1_list = [average] * len(bal_acc1_list)
average = sum(bal_acc3_list) / len(bal_acc3_list)
bal_acc3_list = [average] * len(bal_acc3_list)
average = sum(bal_acc5_list) / len(bal_acc5_list)
bal_acc5_list = [average] * len(bal_acc5_list)
average = sum(bal_acc7_list) / len(bal_acc7_list)
bal_acc7_list = [average] * len(bal_acc7_list)

# print(bal_acc1_list)
# print(bal_acc2_list)
# print(bal_acc3_list)
# print(bal_acc4_list)
# print(bal_acc5_list)
# print(bal_acc6_list)
# print(bal_acc7_list)
plt.plot(ind, bal_acc1_list, label='No Resampling', marker='o')
plt.plot(ind, bal_acc2_list, label='AT-SMOTE', marker='.')
plt.plot(ind, bal_acc3_list, label='SMOTE', marker='.')
plt.plot(ind, bal_acc4_list, label='AT-ROS', marker='v')
plt.plot(ind, bal_acc5_list, label='ROS', marker='v')
plt.plot(ind, bal_acc6_list, label='AT-ADASYN', marker='x')
plt.plot(ind, bal_acc7_list, label='ADASYN', marker='x')
plt.title('CNN',fontsize=18)
plt.xlabel('Accident-triangle weights',fontsize=14)
plt.ylabel('Balanced Accuracy',fontsize=14)
plt.legend(fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=14)
plt.tight_layout()
plt.show()

# clf = LogisticRegression(random_state=RANDOMSTATE)
# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=testrate, random_state=RANDOMSTATE)
# clf.fit(X_train, y_train)
# yyy=abs(clf.predict(X_test)-y_test)
# print(1-sum(yyy)/len(yyy))
# print(balanced_accuracy_score(clf.predict(X_test),y_test))
#
#
# ros = RandomOverSampler(random_state=RANDOMSTATE)
# X_resampled, y_resampled = ros.fit_resample(X, Y,W=None)
# clf = LogisticRegression(random_state=RANDOMSTATE)
# X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=testrate, random_state=RANDOMSTATE)
# clf.fit(X_train, y_train)
# yyy=abs(clf.predict(X_test)-y_test)
# print(1-sum(yyy)/len(yyy))
# print(balanced_accuracy_score(clf.predict(X_test),y_test))
#
#
# ros = RandomOverSampler(random_state=RANDOMSTATE)
# X_resampled, y_resampled = ros.fit_resample(X, Y,W=W)
# print(sorted(Counter(y_resampled).items()))
# clf = LogisticRegression(random_state=RANDOMSTATE)
# X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=testrate, random_state=RANDOMSTATE)
# clf.fit(X_train, y_train)
#
# yyy=abs(clf.predict(X_test)-y_test)
# print(1-sum(yyy)/len(yyy))
# print(balanced_accuracy_score(clf.predict(X_test),y_test))



# clf.fit(X_resampled, y_resampled)
# print(scores)
# df_scores = pd.DataFrame(scores, index=index)
# print(df_scores)
# exit(0)

# 自动生成模型名称和组合
# model_combinations = []
#
# for sampler, classifier in product(samplers, classifiers):
#     model_name = f"{sampler.__class__.__name__} + {classifier.__class__.__name__}"
#     model_combinations.append((model_name, sampler, classifier))
#
# # 创建和评估模型
# for name, sampler, classifier in model_combinations:
#     create_and_evaluate_model(
#         name,
#         create_column_transformer(),
#         sampler,
#         classifier
#     )
#
# df_scores = pd.DataFrame(scores, index=index)
# print(df_scores)

# df_scores.reset_index().to_csv('over.csv', index=False)