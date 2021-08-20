import os
import numpy as np
import math
import gc
import random
import keras
from keras import regularizers
from keras import __version__
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.applications.imagenet_utils import decode_predictions
from keras.models import Model
from keras.layers import Dense, Input, GlobalAveragePooling2D,Dropout,Flatten,Conv1D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.layers.advanced_activations import PReLU
from keras.layers.core import Masking
from keras.preprocessing import image
from sklearn.metrics import roc_curve, auc
from mixpooling import mixpooling
from get_image import get_pat_lab, get_img_lab, pat_to_img, rect_to_mask, add_mask_pat, write_img_pat, sample_balance, divide_for_nfcv

IM_WIDTH, IM_HEIGHT = 299, 299
FC_SIZE = 1024
DIRDICOM_MTT = './MTT/'
DIRDICOM_TTP = './TTP/'

iv3_layer = 0
fold = 4
classes = 2
outpath = './fc_connect_MTT_TTP_outrect{}_layer{}.txt'
cls_dict = {'ca': 1, 'nca': 0, 1: 'ca', 0: 'nca'}

#model
def model_1():
     base_model = InceptionV3( weights = 'imagenet', include_top = False)
     x = base_model.output
     x = GlobalAveragePooling2D(name = 'img_GP1_2')(x)
     feat = Dense(FC_SIZE, activation = 'relu',kernel_regularizer=regularizers.l2(0.0001),name = 'img_dense1_2')(x)
     pred_img = Dense(classes, activation='softmax', name='img_pred_2')(feat)
     model = Model(inputs = base_model.input, outputs = pred_img)
     model.load_weights('./model_weights_img_rect{}fold_layer{}v3_MTT.h5'.format(fold, iv3_layer))                 
     model_img = Model(inputs = base_model.input,outputs = feat)
     return model_img

def model_2():
     base_model = InceptionV3(weights = 'imagenet', include_top = False)
     x = base_model.output
     x = GlobalAveragePooling2D(name = 'img_GP1_3')(x)
     feat = Dense(FC_SIZE, activation = 'relu',kernel_regularizer=regularizers.l2(0.0001),name = 'img_dense1_3')(x)
     pred_img = Dense(classes, activation='softmax', name = 'img_pred_3')(feat)
     model = Model(inputs = base_model.input, outputs = pred_img)
     model.load_weights('./model_weights_img_rect{}fold_layer{}_TTP.h5'.format(fold, iv3_layer))
     model_img = Model(inputs = base_model.input,outputs = feat)
     return model_img

# load the patient images
Patlist_MTT = get_pat_lab(DIRDICOM_MTT, target_size = (IM_WIDTH, IM_HEIGHT))
Patlist_TTP = get_pat_lab(DIRDICOM_TTP, target_size = (IM_WIDTH, IM_HEIGHT))

Patlist_MTT_CA = []
Patlist_MTT_NCA = []
for pat in Patlist_MTT:
        if pat['label'] == 'ca':
                Patlist_MTT_CA.append(pat)
        if pat['label'] == 'nca':
                Patlist_MTT_NCA.append(pat)
random.seed(12)
random.shuffle(Patlist_MTT_CA)
random.shuffle(Patlist_MTT_NCA)


Patlist_TTP_CA = []
Patlist_TTP_NCA = []
for pat in Patlist_TTP:
        if pat['label'] == 'ca':
                Patlist_TTP_CA.append(pat)
        if pat['label'] == 'nca':
                Patlist_TTP_NCA.append(pat)
random.seed(12)
random.shuffle(Patlist_TTP_CA)
random.shuffle(Patlist_TTP_NCA)

Patlist_MTT_CA_train = Patlist_MTT_CA[:int(len(Patlist_MTT_CA) * 0.8)]
Patlist_MTT_CA_test = Patlist_MTT_CA[int(len(Patlist_MTT_CA) * 0.8):]
Patlist_MTT_NCA_train = Patlist_MTT_NCA[:int(len(Patlist_MTT_NCA) * 0.8)]
Patlist_MTT_NCA_test = Patlist_MTT_NCA[int(len(Patlist_MTT_NCA) * 0.8):]

Patlist_TTP_CA_train = Patlist_TTP_CA[:int(len(Patlist_TTP_CA) * 0.8)]
Patlist_TTP_CA_test = Patlist_TTP_CA[int(len(Patlist_TTP_CA) * 0.8):]
Patlist_TTP_NCA_train = Patlist_TTP_NCA[:int(len(Patlist_TTP_NCA) * 0.8)]
Patlist_TTP_NCA_test = Patlist_TTP_NCA[int(len(Patlist_TTP_NCA) * 0.8):]
                                

# patient for train
pat_PWI_trainlist_CA = []
pat_PWI_trainlist_NCA = []
model_feat1 = model_1()
model_feat2 = model_2()

# reshape train data
train_pat_feature_CA = np.zeros((1024,2))
train_pat_feature_NCA = np.zeros((1024,2))

for pat_MTT_CA,pat_TTP_CA in zip(Patlist_MTT_CA_train,Patlist_TTP_CA_train):
        
        label_CA = pat_MTT_CA['label']

        img_list_mtt_CA = pat_MTT_CA['images']
        img_array_mtt_CA = np.array(img_list_mtt_CA).astype('float')/128.0-1.0
        img_feat_mtt_CA = model_feat1.predict(img_array_mtt_CA)
        x_mtt_CA = mixpooling(img_feat_mtt_CA, pooling = 'MAX')

        img_list_ttp_CA = pat_TTP_CA['images']
        img_array_ttp_CA = np.array(img_list_ttp_CA).astype('float')/128.0-1.0
        img_feat_ttp_CA = model_feat2.predict(img_array_ttp_CA)
        x_ttp_CA = mixpooling(img_feat_ttp_CA, pooling = 'MAX')

        train_pat_feature_CA[:,0] = x_mtt_CA
        train_pat_feature_CA[:,1] = x_ttp_CA       
        pat_PWI_CA_info = {'label_ca':label_CA,'pat_ca_feature':train_pat_feature_CA}
        pat_PWI_trainlist_CA.append(pat_PWI_CA_info)


for pat_MTT_NCA,pat_TTP_NCA in zip(Patlist_MTT_NCA_train,Patlist_TTP_NCA_train):
        label_NCA = pat_MTT_NCA['label']

        img_list_mtt_NCA = pat_MTT_NCA['images']
        img_array_mtt_NCA = np.array(img_list_mtt_NCA).astype('float')/128.0-1.0
        img_feat_mtt_NCA = model_feat1.predict(img_array_mtt_NCA)
        x_mtt_NCA = mixpooling(img_feat_mtt_NCA, pooling = 'MAX')

        img_list_ttp_NCA = pat_TTP_NCA['images']
        img_array_ttp_NCA = np.array(img_list_ttp_NCA).astype('float')/128.0-1.0
        img_feat_ttp_NCA = model_feat2.predict(img_array_ttp_NCA)
        x_ttp_NCA = mixpooling(img_feat_ttp_NCA, pooling = 'MAX')

        train_pat_feature_NCA[:,0] = x_mtt_NCA
        train_pat_feature_NCA[:,1] = x_ttp_NCA
        pat_PWI_NCA_info = {'label_nca':label_NCA,'pat_nca_feature':train_pat_feature_NCA}
        pat_PWI_trainlist_NCA.append(pat_PWI_NCA_info)


#balance pat
pat_PWI_trainlist_CA, pat_PWI_trainlist_NCA = sample_balance(pat_PWI_trainlist_CA, pat_PWI_trainlist_NCA, style = 'oversample')

label_list_CA = []
label_list_NCA = []
pat_feature_list_CA = []
pat_feature_list_NCA = []

# get ca_patient for train
for pat_CA in pat_PWI_trainlist_CA:
        label_CA = pat_CA['label_ca']
        label_list_CA.append(cls_dict[label_CA])
        feature_CA = pat_CA['pat_ca_feature']
        pat_feature_list_CA.append(feature_CA)

label_array_CA = np.array(label_list_CA)
pat_array_CA = np.array(pat_feature_list_CA)
x_train_pat_CA = pat_array_CA
y_train_pat_CA = keras.utils.to_categorical(label_array_CA, num_classes = classes)

# get nca_patient for train
for pat_NCA in pat_PWI_trainlist_NCA:
        label_NCA = pat_NCA['label_nca']
        label_list_NCA.append(cls_dict[label_NCA])
        feature_NCA = pat_NCA['pat_nca_feature']
        pat_feature_list_NCA.append(feature_NCA)

label_array_NCA = np.array(label_list_NCA)
pat_array_NCA = np.array(pat_feature_list_NCA)
x_train_pat_NCA = pat_array_NCA
y_train_pat_NCA = keras.utils.to_categorical(label_array_NCA, num_classes = classes)
x_train_pat = np.concatenate((x_train_pat_CA,x_train_pat_NCA),axis = 0)
y_train_pat = np.concatenate((y_train_pat_CA,y_train_pat_NCA),axis = 0)

# patient for test
test_pat_MTT = Patlist_MTT_CA_test + Patlist_MTT_NCA_test
test_pat_TTP = Patlist_TTP_CA_test + Patlist_TTP_NCA_test

pat_mtt_list = []
pat_ttp_list = []
label_list = []
x_test = np.zeros((1024,2))
x_test_list = []
for pat_MTT,pat_TTP in zip(test_pat_MTT,test_pat_TTP):
        label = pat_MTT['label']
        label_list.append(cls_dict[label])
        img_list_mtt = pat_MTT['images']
        img_array_mtt = np.array(img_list_mtt).astype('float') / 128.0 - 1.0
        img_feat_mtt = model_feat1.predict(img_array_mtt)
        x_mtt = mixpooling(img_feat_mtt, pooling = 'MAX')
 
        img_list_ttp = pat_TTP['images']
        img_array_ttp = np.array(img_list_ttp).astype('float') / 128.0 - 1.0
        img_feat_ttp = model_feat2.predict(img_array_ttp)
        x_ttp = mixpooling(img_feat_ttp, pooling = 'MAX')	

        x_test[:,0] = x_mtt
        x_test[:,1] = x_ttp
        x_test_list.append(x_test)

label_array = np.array(label_list)
y_test_pat = keras.utils.to_categorical(label_array, num_classes = classes)
x_test_pat = np.array(x_test_list)
print('x_test_pat_shape')
print(x_test_pat.shape)

#merge model
def modify():  
    input_shape = (1024,2)
    inputs = Input(input_shape)
    x = Conv1D(64,kernel_size = 3 ,strides = 1, activation = 'relu', padding = 'same')(inputs)
    x0 = Conv1D(32,kernel_size = 3 ,strides = 1, activation = 'relu', padding = 'same')(x)
    x1 = Flatten()(x0)
    Dropout(0.4)
    x2 = Dense(1024, activation = 'relu',name = 'pat_dense1_1')(x1)
    x3 = Dense(1024, activation = 'relu',name = 'pat_dense2_1')(x2)
    pred_pat = Dense(classes, activation = 'softmax', name = 'pat_pred')(x3)
    model_pat = Model(inputs = inputs, outputs = pred_pat)
    return model_pat

#编译model
model  = modify()
model.summary()
sgd = SGD(lr = 0.01, decay = 1e-6, momentum = 0.9, nesterov = True)
model.compile(loss = 'categorical_crossentropy', metrics = ['accuracy'],optimizer = sgd)
early_stopping = EarlyStopping(monitor='val_loss', patience=60, verbose=2)
history = model.fit(x_train_pat, y_train_pat, validation_data = (x_test_pat, y_test_pat),batch_size = 64, epochs = 100)

pat_epoch_acc = []
pat_epoch_val_acc = []
pat_epoch_loss = []
pat_epoch_val_loss = []

pat_epoch_acc += history.history['accuracy']
pat_epoch_val_acc += history.history['val_accuracy']
pat_epoch_loss += history.history['loss']
pat_epoch_val_loss += history.history['val_loss']

# ouput the train information
f = open(outpath.format(fold, iv3_layer),'a')
print('pat_epoch_acc:', file = f)
print(pat_epoch_acc, file = f)
print('pat_epoch_val_acc:', file = f)
print(pat_epoch_val_acc, file = f)
print('pat_epoch_loss:', file = f)
print(pat_epoch_loss, file = f)
print('pat_epoch_val_loss:', file = f)
print(pat_epoch_val_loss, file = f)
f.close()

model.save_weights('./model_weights_pat_rect{}fold_layer{}v3_DWI_falir.h5'.format(fold, iv3_layer))

# the ROC for model
preds = model.predict(x_test_pat)
preds_cls = np.argsort(-preds, axis = 1)[:, 0]
label_roc = np.argsort(-y_test_pat, axis = 1)[:, 0]
pred_roc = preds[:, 1]
CP = np.sum(y_test_pat, axis = 0)[1]
CN = np.sum(y_test_pat, axis = 0)[0]
with open(outpath.format(fold, iv3_layer), 'a') as f:
     for label, pred_cls in zip(label_roc, preds_cls):
                print('label:', cls_dict[label], 'predicted class:', cls_dict[pred_cls], file = f)

fpr,tpr,thresholds = roc_curve(np.array(label_roc),np.array(pred_roc),pos_label = 1)
ROC_AUC = auc(fpr,tpr)

f = open(outpath.format(fold, iv3_layer), 'a')
print('plot pat_fpr:',file = f)
print(fpr,file = f)
print('plot pat_tpr:',file = f)
print(tpr,file = f)
print('ROC AUC for pat:', ROC_AUC, file = f)
tpr_and_tnr = tpr + 1 - fpr
index_max = np.argmax(tpr_and_tnr)
threshold_opt = thresholds[index_max]
print('pat_index of max yuedeng:', index_max, 'pat_best threshold is:', threshold_opt, file = f)
TPR = tpr[index_max]
FPR = fpr[index_max]
FP = CN * FPR # false P
TP = CP * TPR # true P
FN = CP - TP # false N
TN = CN - FP # true N
PPV = TP / (TP + FP)
SEN = TPR
SPE = TN / CN
NPV = TN / (TN + FN)
ACC = (TP + TN) / (CP + CN)

print('pat_Accuracy:',ACC, file = f)
print('pat_Sensitivity:', TPR, file = f)
print('pat_Specificity:', SPE, file = f)
print('pat_Positive predictive value:', PPV, file = f)
print('pat_Negative predictive value:', NPV, file = f)
f.close()

#del Filelist_CA_train, Filelist_NCA_train
del x_test_pat, y_test_pat
del x_train_pat, y_train_pat, history
gc.collect()

