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
from keras.layers import Dense, Input, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.layers.core import Masking
from keras.callbacks import EarlyStopping
from keras.preprocessing import image
from sklearn.metrics import roc_curve, auc
from mixpooling import mixpooling
from get_image import get_pat_lab, get_img_lab, pat_to_img, rect_to_mask, add_mask_pat, write_img_pat, sample_balance, divide_for_nfcv

# for n-fold cross-validation
fold_loop = 5 
IM_WIDTH, IM_HEIGHT = 299, 299
FC_SIZE = 1024
DIRDICOM = './dwi'
outpath = './v3_outrect{}_layer{}_DWI2.txt'
cls_dict = {'ca': 1, 'nca': 0, 1: 'ca', 0: 'nca'}
IV3_LAYERS = [0]
classes = 2
batch_img_size = 32
batch_pat_size = 32

# load the patient images and mask images
Patlist = get_pat_lab(DIRDICOM, target_size = (IM_WIDTH, IM_HEIGHT))
print('The number of patients is:', len(Patlist))
datagen = ImageDataGenerator(
	rotation_range = 10,
	width_shift_range = 0.2,
	height_shift_range = 0.2,
	horizontal_flip = False)


Patlist_CA = []
Patlist_NCA = []
for pat in Patlist:
	if pat['label'] == 'ca':
		Patlist_CA.append(pat)
	if pat['label'] == 'nca':
		Patlist_NCA.append(pat)

random.seed(12)
random.shuffle(Patlist_CA)
random.shuffle(Patlist_NCA)
random.seed()

divide_list_pat_CA = divide_for_nfcv(Patlist_CA, group_num = fold_loop, style = 'image')  # style can be 'patient' or 'image'
divide_list_pat_NCA = divide_for_nfcv(Patlist_NCA, group_num = fold_loop, style = 'image')


for iv3_layer in IV3_LAYERS:
	for fold in range(1,n):		
		f = open(outpath.format(fold, iv3_layer),'a')
		print('the first{0} loop for fold cross-validation'.format(fold),file = f)
		f.close()
		p1 = divide_list_pat_CA[fold]
		p2 = divide_list_pat_CA[fold + 1]
		Patlist_CA_train = Patlist_CA[0 : p1] + Patlist_CA[p2 : len(Patlist_CA)]
		Patlist_CA_test = Patlist_CA[p1 : p2]
		p1 = divide_list_pat_NCA[fold]
		p2 = divide_list_pat_NCA[fold + 1]
		Patlist_NCA_train = Patlist_NCA[0 : p1] + Patlist_NCA[p2 : len(Patlist_NCA)]
		Patlist_NCA_test = Patlist_NCA[p1 : p2]
		
		Filelist_CA_train = pat_to_img(Patlist_CA_train)
		Filelist_NCA_train = pat_to_img(Patlist_NCA_train)
		Filelist_CA_test = pat_to_img(Patlist_CA_test)
		Filelist_NCA_test = pat_to_img(Patlist_NCA_test)
		Filelist = pat_to_img(Patlist)	
	
		# generate the model
		base_model = InceptionV3(input_shape = (IM_WIDTH, IM_HEIGHT, 3),weights = 'imagenet', include_top = False) #, input_shape = (229, 229, 3))
		x = base_model.output
		x = GlobalAveragePooling2D(name = 'img_GP1')(x)
		feat = Dense(FC_SIZE, activation = 'relu',kernel_regularizer=regularizers.l2(0.0001),name = 'img_dense1')(x)     
		inputs =  base_model.input
		print(inputs.shape)     
		model_feat = Model(inputs = base_model.input, outputs = feat)
		
		pred_img = Dense(classes, activation='sigmoid', name = 'img_pred')(feat)
		model_img = Model(inputs = base_model.input, outputs = pred_img)
		
		for layer in base_model.layers[:(iv3_layer + 1)]:
			layer.trainable = False
		for layer in base_model.layers[(iv3_layer + 1):]:
			layer.trainable = True

		input_shape = (FC_SIZE,)
		inputs = Input(input_shape)
		x = Dense(FC_SIZE, activation = 'relu',kernel_regularizer=regularizers.l2(0.0001),name = 'pat_dense1')(inputs)
		pred_pat = Dense(classes, activation = 'softmax',name = 'pat_pred')(x)
		model_pat = Model(inputs = inputs, outputs = pred_pat)
		
		sgd = SGD(lr = 0.001, decay = 1e-6, momentum = 0.9, nesterov = True)
		model_img.compile(loss = 'categorical_crossentropy', metrics = ['accuracy'], optimizer = sgd)
		model_pat.compile(loss = 'categorical_crossentropy', metrics = ['accuracy'], optimizer = sgd)
		
		# the number of images for train
		# balance the samples of two classes, style can be set 'oversample' or 'undersample'
		Filelist_CA_train, Filelist_NCA_train = sample_balance( 
			Filelist_CA_train, Filelist_NCA_train, style = 'oversample')
		
		#print('The num_img_ca is:', num_img_CA)
		#print('The num_img_nca is:',num_img_NCA)
		# generate image test
		test_img = Filelist_CA_test + Filelist_NCA_test
		random.shuffle(test_img)
		img_list = []; label_list = []
		for img in test_img:
			x = img['image']
			label = img['label']
			img_list.append(x)
			label_list.append(cls_dict[label])
		img_array = np.array(img_list).astype('float') / 128.0 - 1.0
		#print(img_array.shape)
		label_array = np.array(label_list)
		print(label_array)
		x_test_img = img_array
		y_test_img = keras.utils.to_categorical(label_array, num_classes = classes)
		
	
		
		Patlist_CA_train, Patlist_NCA_train = sample_balance(
			Patlist_CA_train, Patlist_NCA_train, style = 'oversample')
	
		# generate patient test (x_test_pat_list is just a image_array list, not the feature array for model_pat)
		test_pat = Patlist_CA_test + Patlist_NCA_test
		random.shuffle(test_pat)
		pat_list = []; label_list = []
		for pat in test_pat:
			img_list = pat['images']
			label = pat['label']
			label_list.append(cls_dict[label])
			img_array = np.array(img_list).astype('float') / 128.0 - 1.0
			pat_list.append(img_array)
		label_array = np.array(label_list)
		
		x_test_pat_list = pat_list
		y_test_pat = keras.utils.to_categorical(label_array, num_classes = classes)
		
		img_epoch_acc = []
		img_epoch_val_acc = []
		img_epoch_loss = []
		img_epoch_val_loss = []
		pat_epoch_acc = []
		pat_epoch_val_acc = []
		pat_epoch_loss = []
		pat_epoch_val_loss = []
		
		# train img
		train_img_CA = Filelist_CA_train
		train_img_NCA = Filelist_NCA_train
		train_img = train_img_CA + train_img_NCA
		random.shuffle(train_img)
		img_list = []
		label_list = []
		for img in train_img:
			x = img['image']
			label = img['label']
			img_list.append(x)
			label_list.append(cls_dict[label])
		img_array = np.array(img_list).astype('float') / 128.0 - 1.0
		label_array = np.array(label_list)
		
		x_train_img = img_array
		print(x_train_img.shape)
		y_train_img = keras.utils.to_categorical(label_array, num_classes = classes)

		early_stopping = EarlyStopping(monitor='val_loss', patience=100, verbose=2)		

		history = model_img.fit_generator(
			datagen.flow(x_train_img, y_train_img, batch_size = batch_img_size),
			validation_data = (x_test_img, y_test_img),
			epochs = 300,
			verbose=2, shuffle=False, callbacks=[early_stopping]) # epochs = 150
		
		num_x_train_img = len(x_train_img)
		print('x_tain_img:',num_x_train_img)
		img_epoch_acc += history.history['accuracy']
		img_epoch_val_acc += history.history['val_accuracy']
		img_epoch_loss += history.history['loss']
		img_epoch_val_loss += history.history['val_loss']
		
		# train patient
		train_pat_CA = Patlist_CA_train
		train_pat_NCA = Patlist_NCA_train
		train_pat = train_pat_CA + train_pat_NCA
		random.shuffle(train_pat)
		pat_list = []
		label_list = []
		for pat in train_pat:
			img_list = pat['images']
			label = pat['label']
			label_list.append(cls_dict[label])
			img_array = np.array(img_list).astype('float') / 128.0 - 1.0
			img_feat = model_feat.predict(img_array)
			x = mixpooling(img_feat, pooling = 'MAX')
			pat_list.append(x)
		
		pat_array = np.array(pat_list)	
		label_array = np.array(label_list)
		x_train_pat = pat_array
		y_train_pat = keras.utils.to_categorical(label_array, num_classes = classes)
	
		# generate the features for test
		pat_list = []
		for img_array in x_test_pat_list:
			img_feat = model_feat.predict(img_array)
			x = mixpooling(img_feat, pooling = 'MAX')
			pat_list.append(x)
		x_test_pat = np.array(pat_list)
		
		early_stopping = EarlyStopping(monitor='val_loss', patience=100, verbose=2)

		history = model_pat.fit(x_train_pat, y_train_pat, validation_data = (x_test_pat, y_test_pat),
			epochs=300, batch_size = batch_pat_size, verbose=2, shuffle=False, callbacks=[early_stopping]) # epochs = 500
		pat_epoch_acc += history.history['accuracy']
		pat_epoch_val_acc += history.history['val_accuracy']
		pat_epoch_loss += history.history['loss']
		pat_epoch_val_loss += history.history['val_loss']
			
		# ouput the train information
		f = open(outpath.format(fold, iv3_layer),'a')
		print('img_epoch_acc:',file = f)
		print(img_epoch_acc,file = f)
		print('img_epoch_val_acc:', file = f)
		print(img_epoch_val_acc,file = f)
		print('img_epoch_loss:', file = f)
		print(img_epoch_loss,file = f)
		print('img_epoch_val_loss', file = f)
		print(img_epoch_val_loss,file = f)
	
		print('pat_epoch_acc:', file = f)
		print(pat_epoch_acc, file = f)
		print('pat_epoch_val_acc:', file = f)
		print(pat_epoch_val_acc, file = f)
		print('pat_epoch_loss:', file = f)
		print(pat_epoch_loss, file = f)
		print('pat_epoch_val_loss:', file = f)
		print(pat_epoch_val_loss, file = f)
		f.close()
		
		# save model 
		model_img.save_weights('./model_weights_img_rect{}fold_layer{}v3_DWI.h5'.format(fold, iv3_layer))
		model_pat.save_weights('./model_weights_pat_rect{}fold_layer{}v3_DWI.h5'.format(fold, iv3_layer))
		
		
		# the ROC for model_img
		preds = model_img.predict(x_test_img)
		preds_cls = np.argsort(-preds, axis = 1)[:, 0]
		label_roc = np.argsort(-y_test_img, axis = 1)[:, 0]
		pred_roc = preds[:, 1]
		CP = np.sum(y_test_img, axis = 0)[1]
		CN = np.sum(y_test_img, axis = 0)[0]
		with open(outpath.format(fold, iv3_layer),  'a') as f:
			for label, pred_cls in zip(label_roc, preds_cls):
				print('label:', cls_dict[label], 'predicted class:', cls_dict[pred_cls], file = f)
	
		fpr, tpr, thresholds = roc_curve(np.array(label_roc),np.array(pred_roc),pos_label = 1)
		ROC_AUC = auc(fpr,tpr)
		
		f = open(outpath.format(fold, iv3_layer),'a')
		print('plot img_fpr:',file = f)
		print(fpr,file = f)
		print('plot img_tpr:',file = f)
		print(tpr,file = f)
		print('ROC AUC for img:', ROC_AUC, file = f)
		tpr_and_tnr = tpr + 1 - fpr
		index_max = np.argmax(tpr_and_tnr)
		threshold_opt = thresholds[index_max]
		print('img_index of max yuedeng:', index_max, 'img_best threshold is:', threshold_opt, file = f)
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
		MCC = (TP * TN - FP * FN) / math.sqrt(CP * CN * (TP + FP) * (FN + TN))
		print('img_Accuracy:',ACC, file = f)
		print('img_Sensitivity:', TPR, file = f)
		print('img_Specificity:', SPE, file = f)
		print('img_Positive predictive value:', PPV, file = f)
		print('img_Negative predictive value:', NPV, file = f)
		print('img_MCC:', MCC, file = f)
		f.close()
		
	
		# the ROC for model_pat
		pat_list = []
		for img_array in x_test_pat_list:
			img_feat = model_feat.predict(img_array)
			x = mixpooling(img_feat, pooling = 'MAX')
			pat_list.append(x)
		x_test_pat = np.array(pat_list)
		
		preds = model_pat.predict(x_test_pat)
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
		MCC = (TP * TN - FP * FN) / math.sqrt(CP * CN * (TP + FP) * (FN + TN))
		print('pat_Accuracy:',ACC, file = f)
		print('pat_Sensitivity:', TPR, file = f)
		print('pat_Specificity:', SPE, file = f)
		print('pat_Positive predictive value:', PPV, file = f)
		print('pat_Negative predictive value:', NPV, file = f)
		print('pat_MCC:', MCC, file = f)
		f.close()
	
		del base_model, feat, model_feat, pred_img, model_img, pred_pat, model_pat
		del Filelist_CA_train, Filelist_NCA_train, Patlist_CA_train, Patlist_NCA_train
		del x_test_img, y_test_img, x_test_pat_list, x_test_pat, y_test_pat
		del x_train_img, y_train_img, x_train_pat, y_train_pat, history
		gc.collect()
		
		
