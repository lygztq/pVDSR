import numpy as np
from scipy import misc
from PIL import Image
import tensorflow as tf
import glob, os, re
from psnr import psnr
import scipy.io
import pickle
from model import model
import time
import hyper
DATA_PATH = hyper.TEST_DATA_PATH
MAX_EPOCH = hyper.MAX_EPOCH

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--model_path")
args = parser.parse_args()
model_path = args.model_path


def get_img_list(data_path):
	l = glob.glob(os.path.join(data_path,"*"))
	l = [f for f in l if re.search("^\d+.mat$", os.path.basename(f))]
	train_list = []
	for f in l:
		if os.path.exists(f):
			if os.path.exists(f[:-4]+"_2.mat"): train_list.append([f, f[:-4]+"_2.mat", 2])
			if os.path.exists(f[:-4]+"_3.mat"): train_list.append([f, f[:-4]+"_3.mat", 3])
			if os.path.exists(f[:-4]+"_4.mat"): train_list.append([f, f[:-4]+"_4.mat", 4])
	return train_list


def get_test_image(test_list, offset, batch_size):
	target_list = test_list[offset:offset+batch_size]
	input_list = []
	gt_list = []
	scale_list = []
	for pair in target_list:
		print pair[1]
		mat_dict = scipy.io.loadmat(pair[1])
		input_img = None
		if mat_dict.has_key("img_2"): 	input_img = mat_dict["img_2"]
		elif mat_dict.has_key("img_3"): input_img = mat_dict["img_3"]
		elif mat_dict.has_key("img_4"): input_img = mat_dict["img_4"]
		else: continue
		gt_img = scipy.io.loadmat(pair[0])['img_raw']
		input_list.append(input_img)
		gt_list.append(gt_img)
		scale_list.append(pair[2])
	return input_list, gt_list, scale_list


def test_pVDSR_with_sess(ckpt_path, data_path, sess):
	folder_list = glob.glob(os.path.join(data_path, 'Set*'))
	print 'folder_list', folder_list
	if not os.path.exists('./output_img'):
		os.mkdir('./output_img')
	
	psnr_dict = {}
	for folder_path in folder_list:
		psnr_list = []
		img_list = get_img_list(folder_path)
		for i in range(len(img_list)):
			input_list, gt_list, scale_list = get_test_image(img_list, i, 1)
			input_y = input_list[0]
			gt_y = gt_list[0]
			start_t = time.time()
			img_vdsr_y = sess.run([output_tensor], feed_dict={input_tensor: np.resize(input_y, (1, input_y.shape[0], input_y.shape[1], 1))})
			img_vdsr_y = np.resize(img_vdsr_y, (input_y.shape[0], input_y.shape[1]))
			end_t = time.time()

			# save test image and results
			id = img_list[i][1].split('/')[-1].split('.')[0]
			output_id = 'output' + id + '.png'
			input_id = 'input' + id + '.png'
			misc.imsave(os.path.join('./output_img',output_id),img_vdsr_y)
			misc.imsave(os.path.join('./output_img',input_id),input_y)

			print "end_t",end_t,"start_t",start_t
			print "time consumption",end_t-start_t
			print "image_size", input_y.shape
			
			# calculate psnr
			psnr_bicub = psnr(input_y, gt_y, scale_list[0])
			psnr_vdsr = psnr(img_vdsr_y, gt_y, scale_list[0])
			print "PSNR: bicubic %f\tpVDSR %f" % (psnr_bicub, psnr_vdsr)
			psnr_list.append([psnr_bicub, psnr_vdsr, scale_list[0]])
		psnr_dict[os.path.basename(folder_path)] = psnr_list

	if not os.path.exists('./psnr'):
		os.mkdir('psnr')
	with open('psnr/%s' % "result", 'wb') as f:
		pickle.dump(psnr_dict, f)


if __name__ == '__main__':
	ckpt_dir = './checkpoints'
	with tf.Session() as sess:
		input_tensor = tf.placeholder(tf.float32, shape=(1, None, None, 1))
		shared_model = tf.make_template('shared_model', model)
		output_tensor, weights = shared_model(input_tensor)
		init = tf.global_variables_initializer()
		sess.run(init)
		ckpt = tf.train.get_checkpoint_state(ckpt_dir)
		saver = tf.train.Saver(weights)
		saver.restore(sess, tf.train.latest_checkpoint(ckpt_dir))
		test_pVDSR_with_sess(ckpt_dir, DATA_PATH, sess)
