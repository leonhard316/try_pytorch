from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

y_true_multi = [1, 3, 2, 3, 0, 3, 3, 1, 1, 2, 3, 1, 3, 1, 2, 3]
# alexnet
#y_pred_multi = [0, 2, 3, 2, 3, 2, 2, 0, 1, 0, 2, 1, 1, 3, 1, 3]
# vgg
#y_pred_multi = [0, 3, 2, 3, 0, 3, 3, 2, 0, 2, 3, 0, 3, 2, 1, 3]

# データ拡張する前
#y_pred_multi = [0, 2, 3, 2, 1, 1, 2, 0, 0, 1, 2, 1, 2, 0, 1, 2]
# resnet
y_pred_multi = [1, 2, 1, 2, 1, 2, 2, 1, 1, 0, 2, 1, 3, 0, 1, 2]

print(confusion_matrix(y_true_multi, y_pred_multi))

# Accuracy
acc_score = accuracy_score(y_true_multi, y_pred_multi)
print(acc_score)

# Recall
rec_score = recall_score(y_true_multi, y_pred_multi, average=None)
print(rec_score)

# Precision
pre_score = precision_score(y_true_multi, y_pred_multi, average=None)
print(pre_score)

# F値
from sklearn.metrics import f1_score
f1_score = f1_score(y_true_multi, y_pred_multi, average=None)
print(f1_score)