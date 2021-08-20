import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET

from keras.preprocessing import image

def rect_to_mask(Rectlist, target_size = (229, 229), dist = 0):
	Masklist = []
	if 'filepaths' in Rectlist[0].keys(): # the list is patient
		for pat in Rectlist:
			patname = pat['name']
			labname = pat['label']
			filepaths = pat['filepaths']
			rects = pat['rects']
			imgs = []
			mask = {'name': patname, 'label': labname, 'filepaths': filepaths}
			for rect_info in rects:
				[x1, y1, x2, y2] = extand(rect_info['rect'], size = rect_info['size'], dist = dist)
				(h, w, d) = rect_info['size']
				(tg_h, tg_w) = target_size
				tg_x1 = int(x1 * tg_h // h); tg_x2 = int(x2 * tg_h // h);
				tg_y1 = int(y1 * tg_w // w); tg_y2 = int(y2 * tg_w // w);
				img = np.zeros((tg_h, tg_w, d))
				img[tg_x1:tg_x2 + 1, tg_y1:tg_y2 + 1, :] = 255
				imgs.append(img)
			mask['images'] = imgs
			Masklist.append(mask)
	return Masklist

def extand(rect, size, dist = 0):
	[x1, y1, x2, y2] = rect
	(h, w, depth) = size
	if -1 < dist < 1:
		dx = dist * (x2 - x1)
		dy = dist * (y2 - y1)
	else: 
		dx = dist
		dy = dist 
	x1 = max(0, x1 - dx)
	y1 = max(0, y1 - dy)
	x2 = min(h - 1, x2 + dx)
	y2 = min(w - 1, y2 + dy)
	return [x1, y1, x2, y2]

def add_mask_pat(Patlist, Masklist):
	PatROIlist = []
	for pat1 in Patlist:
		patname1 = pat1['name']
		labname1 = pat1['label']
		filepaths1 = pat1['filepaths']
		imgs1 = pat1['images']
		rois = []
		roifilepaths = []
		for j, pat2 in enumerate(Masklist):
			patname2 = pat2['name']
			labname2 = pat2['label']
			filepaths2 = pat2['filepaths']
			imgs2 = pat2['images']
			if labname1 == labname2 and patname1 == patname2:
				for filepath1, img1 in zip(filepaths1, imgs1):
					pathlist1 = filepath1.strip().split('/')
					filename1 = pathlist1[-1].replace('.jpg', '')
					for i, filepath2, img2 in zip(range(len(imgs2)), filepaths2, imgs2):
						pathlist2 = filepath2.strip().split('/')
						filename2 = pathlist2[-1].replace('.xml', '')
						#filepath_img = filepath2.replace('.xml', '.jpg')
						if filename1 in filename2:
							# Masklist[j]['filepaths'][i] = ' ' # for check
							mask = img2 > 6	
							roi = img1 * mask
							rois.append(roi)
							roifilepaths.append(filepath2)
							#print(filepath_img, '*****')
							'''img3 = np.zeros(img1.shape)
							img3[:, :, 0] = img1[:, :, 2]
							img3[:, :, 1] = img1[:, :, 1]
							img3[:, :, 2] = img1[:, :, 0]
							cv2.imwrite(filepath_img, roi)'''

		if rois != []:
			patroi = {'name': patname1, 'label': labname1, 'filepaths': roifilepaths, 'images': rois}
			PatROIlist.append(patroi)
	'''
	# for check the missing mask image
	for pat2 in Masklist:
		filepaths2 = pat2['filepaths']
		for filepath2 in filepaths2:
			if filepath2 != ' ':
				print(filepath2)
	'''

	return PatROIlist


def get_pat_lab(dir_image = None, target_size = (229, 229), filestyle = '.jpg'):

	if dir_image == None:
		print('Please give the directory for patient')
		return None
	Patlist = []
	for dirname, subdirlist, filelist in os.walk(dir_image):
		if len(filelist) != 0:
			dirlist = dirname.strip().split('/')
			patname = dirlist[3]
			labname = dirlist[2]
			filepaths = []
			if filestyle == '.jpg':
				imgs = []
				for filename in filelist:
					filepath = os.path.join(dirname, filename)
					if '.jpg' in filename.lower(): # check whether the file is  jpg
						img = load_img(filepath, target_size)
						imgs.append(img)
						filepaths.append(filepath)
			
				if filepaths == []:
					continue
				patinfo = {'name': patname, 'label': labname, 'filepaths': filepaths, 'images': imgs}
				Patlist.append(patinfo)
			elif filestyle == '.xml':
				rects = []
				for filename in filelist:
					filepath = os.path.join(dirname, filename)
					#print(filepath)
					if filestyle in filename.lower(): # check whether the file is xml
						xml_info = load_xml(filepath)
						xml_filename = xml_info['filename']
						xml_pat = xml_info['name']
						xml_label = xml_info['label']
						xml_rects = xml_info['rects']
						rects += xml_rects
						for i in range(len(xml_rects)):
							fp = filepath
							if i > 0:
								fp = fp[:-4] + '_' + str(i) +fp[-4:]
							filepaths.append(fp)
						
				if filepaths == []:
					continue
				patinfo = {'name':patname, 'label': labname, 'filepaths': filepaths, 'rects': rects}
				Patlist.append(patinfo)
	Patlist = sorted(Patlist, key = lambda e: e.__getitem__('name'))
	return Patlist

def get_img_lab(dir_image = None, target_size = (229, 229)):

	if dir_image == None:
		print('Please give the directory for image')
		return None
	Filelist = []
	for dirname, subdirlist, filelist in os.walk(dir_image):
		dirlist = dirname.strip().split('/')
		labname = dirlist[2]
		for filename in filelist:
			if '.jpg' in filename.lower(): # check whether the file is jpg
				filepath = os.path.join(dirname, filename)
				img = load_img(filepath, target_size)
				fileinfo = {'label': labname, 'filepath': filepath, 'image': img}
				Filelist.append(fileinfo)
	Filelist = sorted(Filelist, key = lambda e: e.__getitem__('name'))
	return Filelist


def pat_to_img(Patlist):
	Filelist = []
	for patinfo in Patlist:
		filepaths = patinfo['filepaths']
		labname = patinfo['label']
		if 'images' in patinfo.keys():
			imgs = patinfo['images']
			for filepath, img in zip(filepaths, imgs):
				fileinfo = {'label': labname, 'filepath': filepath, 'image': img}
				Filelist.append(fileinfo)
		elif 'rects' in patinfo.keys():
			rects = patinfo['rects']
			for filepath, rect in zip(filepaths, rects):
				fileinfo = {'label': labname, 'filepath': filepath, 'rect': rect}
				Filelist.append(fileinfo)
	Filelist = sorted(Filelist, key = lambda e: e.__getitem__('filepath'))
	return Filelist


def sample_balance(samples1, samples2, style = 'oversample'):
	num1 = len(samples1)
	num2 = len(samples2)
	if num1 < num2:
		if style == 'oversample':
			samples1 += np.random.choice(
				samples1, num2 - num1, replace = True).tolist()	
		elif style == 'undersample':
			samples2 = np.random.choice(
				samples2, num1, replace = False).tolist()
	
	elif num1 > num2:
		if style == 'oversample':
			samples2 += np.random.choice(
				samples2, num1 - num2, replace = True).tolist()
		elif style == 'undersample':
			samples1 = np.random.choice(
				samples1, num2, replace = True).tolist()
	num1 = len(samples1)
	num2 = len(samples2)
	print(num1, num2)
	return samples1, samples2



def load_img(filepath, target_size = (229, 229)):
	width_height_tuple = (target_size[1], target_size[0])
	I = image.load_img(filepath)
	I = I.resize(width_height_tuple)
	img = image.img_to_array(I)
	return img

def load_xml(filepath):
	et = ET.parse(filepath)
	element = et.getroot()
	element_objs = element.findall('object')
	element_label = element.find('label').text
	element_patient = element.find('patient').text
	element_filename = element.find('filename').text
	element_width = int(element.find('size').find('width').text)
	element_height = int(element.find('size').find('height').text)
	element_depth = int(element.find('size').find('depth').text)
	size = [element_height, element_width, element_depth]
	
	rects_info = []
	if len(element_objs) > 0:
		for element_obj in element_objs:
			class_name = element_obj.find('name').text
			obj_bbox = element_obj.find('bndbox')
			# the order of x and y is important, we changed the order here,
			# because of x is for height, and y is for width in our code
			y1 = int(round(float(obj_bbox.find('xmin').text)))
			x1 = int(round(float(obj_bbox.find('ymin').text)))
			y2 = int(round(float(obj_bbox.find('xmax').text)))
			x2 = int(round(float(obj_bbox.find('ymax').text)))
			rect = [x1, y1, x2, y2]
			rect_info = {'class_name': class_name, 'rect': rect, 'size': size}
			rects_info.append(rect_info)
	xml = {'label': element_label, 'name': element_patient, 'filename': element_filename,
		'rects': rects_info}
	return xml

def write_img_pat(Patlist, DIR = ''):
	for pat in Patlist:
		filepaths = pat['filepaths']
		imgs = pat['images']
		for filepath, img in zip(filepaths, imgs):
			pathlist = filepath.strip().split('/')
			if DIR != '':
				filepath = filepath.replace(pathlist[0] + '/', DIR)
			filepath = filepath[:-4] + '.jpg'
			if check_path(filepath):
				cv2.imwrite(filepath, img)
			
def check_path(filepath):
	pathlist = filepath.strip().split('/')
	path = pathlist[0]
	pathlist = pathlist[1:-1]
	for pi in pathlist:
		path = os.path.join(path, pi)
		#print(path)
		if os.path.exists(path):
			continue
		os.makedirs(path)
	return True


def divide_for_nfcv(Patlist, group_num = 5, style = 'patient'): # style can be 'patient' or 'image'
	divide_list_pat = []
	
	if style == 'patient':
		num_pat = len(Patlist)
		for i in range(group_num + 1):
			p_pat = round(num_pat * float(i) / group_num)
		divide_list_pat.append(p_pat)
	elif style == 'image':
		num_img = len(pat_to_img(Patlist))
		num_img_list = []
		for pat in Patlist:
			print(pat['name'])
			num = len(pat['images'])
			num_img_list.append(num)
			
		divide_list_img = []
		for i in range(group_num + 1):
			p_img = round(num_img * float(i) / group_num)
			divide_list_img.append(p_img)

		print(num_img_list)
		print(divide_list_img)
		num_img = 0
		group_index = 1
		divide_list_pat.append(0)
		for i, num in enumerate(num_img_list):
			if group_index == group_num:
				divide_list_pat.append(len(num_img_list))
				print(num_img_list)
				break
			
			if num_img + num >= divide_list_img[group_index]:
			
				if num / 2 < (divide_list_img[group_index] - num_img):
					divide_list_pat.append(i + 1)
				else:
					divide_list_pat.append(i)
				group_index += 1
				print(num_img_list)
			num_img += num
			print('------------------')
	return divide_list_pat
