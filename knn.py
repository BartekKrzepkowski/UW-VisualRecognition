# k=1:  L1=<acc_val/acc_test>  L2=<acc_val/acc_test>
# k=3:  L1=<acc_val/acc_test>  L2=<acc_val/acc_test>
# k=5:  L1=<acc_val/acc_test>  L2=<acc_val/acc_test>
# k=7:  L1=<acc_val/acc_test>  L2=<acc_val/acc_test>
# for L1 choose k=<1/3/5/7> on <val/test> set
# for L2 choose k=<1,3,5,7> on <val/test> set

from cifar10 import load_CIFAR10
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

def load_data():
    Xtr, Ytr, Xte, Yte = load_CIFAR10('cifar-10-batches-py')
    Xtr = np.array(Xtr,dtype=np.int32)
    Ytr = np.array(Ytr,dtype=np.int32)
    Xte = np.array(Xte,dtype=np.int32)
    Yte = np.array(Yte,dtype=np.int32)
    Xva = Xtr[40001:,:]
    Yva = Ytr[40001:]
    Xtr = Xtr[:40000,:]
    Ytr = Ytr[:40000]
    indices = range(0,Xte.shape[0],200)
    return Xtr, Ytr, Xva[indices,:], Yva[indices], Xte[indices,:], Yte[indices]


Xtr, Ytr, Xva, Yva, Xte, Yte = load_data()
ntest = Xte.shape[0]

print "Nearest Neighbor"
pred_nn_va_l1 = ...  # TODO
pred_nn_va_l2 = ...  # TODO
pred_nn_te_l1 = ...  # TODO
pred_nn_te_l2 = ...  # TODO
print "Accuracy L1:", accuracy(pred_nn_va_l1), accuracy(pred_nn_te_l1)  # TODO
print confusion_matrix(pred_nn_te_l1)  # TODO
print "Accuracy L2:", accuracy(pred_nn_va_l2), accuracy(pred_nn_te_l2)  # TODO
print confusion_matrix(pred_nn_te_l2)  # TODO

for k in [1,3,5,7]:
    print "k-NearestNeighbors for k =",k
    KNeighborsClassifier ...  # TODO
    pred_knn_va_l1 = ...  # TODO
    pred_knn_te_l1 = ...  # TODO
    KNeighborsClassifier ...  # TODO
    pred_knn_va_l2 = ...  # TODO
    pred_knn_te_l2 = ...  # TODO
    print "Accuracy L1:",  # TODO (as above)
    print confusion_matrix(pred_knn_te_l1)  # TODO
    print "Accuracy L2:",  # TODO (as above)
    print confusion_matrix(pred_knn_te_l2)  # TODO

