import os, glob, re, signal, sys, argparse, threading, time
from random import shuffle
import random
import tensorflow as tf
from PIL import Image
import numpy as np
import scipy.io
from model import model
from psnr import psnr
from test import test_pVDSR
import hyper

# Get hyper-parameters
DATA_PATH = hyper.DATA_PATH
IMG_SIZE = hyper.IMG_SIZE
BATCH_SIZE = hyper.BATCH_SIZE
BASE_LR = hyper.BASE_LR
LR_RATE = hyper.LR_RATE
LR_STEP_SIZE = hyper.LR_STEP_SIZE
MAX_EPOCH = hyper.MAX_EPOCH
USE_QUEUE_LOADING = hyper.USE_QUEUE_LOADING
TEST_DATA_PATH = hyper.TEST_DATA_PATH

# the check point path
parser = argparse.ArgumentParser()
parser.add_argument("--model_path")
args = parser.parse_args()
model_path = args.model_path

#def get_hyper(using_default=True):
#	'''
#	Get the hyper-parameter
#	'''
 #       global DATA_PATH, IMG_SIZE, BATCH_SIZE, BASE_LR, LR_RATE, LR_STEP_SIZE, MAX_EPOCH, USE_QUEUE_LOADING, TEST_DATA_PATH
#	if using_default:
#		return
#	else:
#		DATA_PATH = hyper['DATA_PATH']
#		IMG_SIZE = hyper['IMG_SIZE']
#		BATCH_SIZE = hyper['BATCH_SIZE']
#		BASE_LR = hyper['BASE_LR']
#		LR_RATE = hyper['LR_RATE']
#		LR_STEP_SIZE = hyper['LR_STEP_SIZE']
#		MAX_EPOCH = hyper['MAX_EPOCH']
#		USE_QUEUE_LOADING = hyper['USE_QUEUE_LOADING']
#		TEST_DATA_PATH = hyper['TEST_DATA_PATH']

def get_train_list(data_path):
	'''
	Get training data list
	'''
	mat_list = glob.glob(os.path.join(data_path,"*"))
	mat_list = [f for f in mat_list if re.search("^\d+.mat$", os.path.basename(f))]
	print "Number of training data: %d\n" % (4*len(mat_list))
	train_list = []
	for mat in mat_list:
		if os.path.exists(mat):
			for i in range(2,5):
				back_name = "_%d.mat" % i
				if os.path.exists(mat[:-4]+back_name): 
					train_list.append([mat,mat[:-4]+back_name])
			# if os.path.exists(f[:-4]+"_2.mat"): train_list.append([f, f[:-4]+"_2.mat"])
			# if os.path.exists(f[:-4]+"_3.mat"): train_list.append([f, f[:-4]+"_3.mat"])
			# if os.path.exists(f[:-4]+"_4.mat"): train_list.append([f, f[:-4]+"_4.mat"])
	return train_list

def get_image_batch(train_list,offset,batch_size):
	'''
	Get a batch of image
	'''
	target_list = train_list[offset:offset+batch_size]
	input_list = []
	label_list = []
	cbcr_list = []
	for pair in target_list:
		input_img = scipy.io.loadmat(pair[1])['patch']
		label_img = scipy.io.loadmat(pair[0])['patch']
		input_list.append(input_img)
		label_list.append(label_img)
	input_list = np.array(input_list)
	input_list.resize([BATCH_SIZE, IMG_SIZE[1], IMG_SIZE[0], 1])
	label_list = np.array(label_list)
	label_list.resize([BATCH_SIZE, IMG_SIZE[1], IMG_SIZE[0], 1])
	return input_list, label_list, np.array(cbcr_list)

def get_test_image(test_list, offset, batch_size):
	'''
	Get a batch of test image
	'''
	target_list = test_list[offset:offset+batch_size]
	input_list = []
	label_list = []
	for pair in target_list:
		mat_dict = scipy.io.loadmat(pair[1])
		input_img = None
		if mat_dict.has_key("img_2"): 	input_img = mat_dict["img_2"]
		elif mat_dict.has_key("img_3"): input_img = mat_dict["img_3"]
		elif mat_dict.has_key("img_4"): input_img = mat_dict["img_4"]
		else: continue
		label_img = scipy.io.loadmat(pair[0])['img_raw']
		input_list.append(input_img[:,:,0])
		label_list.append(label_img[:,:,0])
	return input_list, label_list

def main():
	# Get input data
	#get_hyper(using_default=True)
	train_list = get_train_list(DATA_PATH)
	
	if not USE_QUEUE_LOADING: # without asynchronous data loading
		print "Not use queue loading"
		train_input  	= tf.placeholder(tf.float32, shape=(BATCH_SIZE, IMG_SIZE[0], IMG_SIZE[1], 1))
		train_gt  		= tf.placeholder(tf.float32, shape=(BATCH_SIZE, IMG_SIZE[0], IMG_SIZE[1], 1))

	else:
		print "Use queue loading"	# with asynchronous data loading
		train_input_single  = tf.placeholder(tf.float32, shape=(IMG_SIZE[0], IMG_SIZE[1], 1))
		train_gt_single  	= tf.placeholder(tf.float32, shape=(IMG_SIZE[0], IMG_SIZE[1], 1))
		q = tf.FIFOQueue(10000, [tf.float32, tf.float32], [[IMG_SIZE[0], IMG_SIZE[1], 1], [IMG_SIZE[0], IMG_SIZE[1], 1]])
		enqueue_op = q.enqueue([train_input_single, train_gt_single])
    
		train_input, train_gt	= q.dequeue_many(BATCH_SIZE)




	# Define model and training paramaters

	# 	model
	shared_model = tf.make_template('shared_model', model)
	train_output, weights 	= shared_model(train_input)
	loss = tf.reduce_sum(tf.nn.l2_loss(tf.subtract(train_output, train_gt)))	# MSE
	for w in weights:
		loss += tf.nn.l2_loss(w)*1e-4
	tf.summary.scalar("loss", loss)

	# 	learning step and optimizer
	global_step 	= tf.Variable(0, trainable=False)
	learning_rate 	= tf.train.exponential_decay(BASE_LR, global_step*BATCH_SIZE, len(train_list)*LR_STEP_SIZE, LR_RATE, staircase=True)
	tf.summary.scalar("learning rate", learning_rate)

	optimizer = tf.train.AdamOptimizer(learning_rate)#tf.train.MomentumOptimizer(learning_rate, 0.9)
	opt = optimizer.minimize(loss, global_step=global_step)

	saver = tf.train.Saver(weights, max_to_keep=0)

	shuffle(train_list)
	config = tf.ConfigProto()




	# Training initialization

	with tf.Session(config=config) as sess:
		#TensorBoard open log with "tensorboard --logdir=logs"
		if not os.path.exists('logs'):
			os.mkdir('logs')
		merged = tf.summary.merge_all()
		file_writer = tf.summary.FileWriter('logs', sess.graph)
                
                init = tf.global_variables_initializer()
                sess.run(init)
		#tf.initialize_all_variables().run()

		if model_path:
			print "restore model..."
			saver.restore(sess, model_path)
			print "Done"

		# Functions using for queue training
		def load_and_enqueue(coord, file_list, enqueue_op, train_input_single, train_gt_single, idx=0, num_thread=1):
			count = 0;
			length = len(file_list)
			try:
				while not coord.should_stop():
					i = count % length;
					input_img	= scipy.io.loadmat(file_list[i][1])['patch'].reshape([IMG_SIZE[0], IMG_SIZE[1], 1])
					label_img		= scipy.io.loadmat(file_list[i][0])['patch'].reshape([IMG_SIZE[0], IMG_SIZE[1], 1])
					sess.run(enqueue_op, feed_dict={train_input_single:input_img, train_gt_single:label_img})
					count+=1
			except Exception as e:
				print "stopping...", idx, e

		threads = []
		def signal_handler(signum,frame):
			sess.run(q.close(cancel_pending_enqueues=True))
			coord.request_stop()
			coord.join(threads)
			print "Done"
			sys.exit(1)
		original_sigint = signal.getsignal(signal.SIGINT)
		signal.signal(signal.SIGINT, signal_handler)


		# Start training

		if USE_QUEUE_LOADING:
			# create threads
			num_thread=20
			coord = tf.train.Coordinator()
			for i in range(num_thread):
				length = len(train_list)/num_thread
				t = threading.Thread(target=load_and_enqueue, args=(coord, train_list[i*length:(i+1)*length],enqueue_op, train_input_single, train_gt_single,  i, num_thread))
				threads.append(t)
				t.start()
			print "num thread:" , len(threads)

			for epoch in xrange(0, MAX_EPOCH):
				max_step=len(train_list)//BATCH_SIZE
				for step in range(max_step):
					_,l,output,lr, g_step, summary = sess.run([opt, loss, train_output, learning_rate, global_step, merged])
					print "[epoch %2.4f] loss %.4f\t lr %.5f"%(epoch+(float(step)*BATCH_SIZE/len(train_list)), np.sum(l)/BATCH_SIZE, lr)
					file_writer.add_summary(summary, step+epoch*max_step)
					#print "[epoch %2.4f] loss %.4f\t lr %.5f\t norm %.2f"%(epoch+(float(step)*BATCH_SIZE/len(train_list)), np.sum(l)/BATCH_SIZE, lr, norm)
				saver.save(sess, "./checkpoints/pVDSR_adam_epoch_%03d.ckpt" % epoch ,global_step=global_step)
		else:
			for epoch in xrange(0, MAX_EPOCH):
				for step in range(len(train_list)//BATCH_SIZE):
					offset = step*BATCH_SIZE
					input_data, gt_data,cbcr_data = get_image_batch(train_list, offset, BATCH_SIZE) #, cbcr_data
					feed_dict = {train_input: input_data, train_gt: gt_data}
					_,l,output,lr, g_step = sess.run([opt, loss, train_output, learning_rate, global_step], feed_dict=feed_dict)
					print "[epoch %2.4f] loss %.4f\t lr %.5f"%(epoch+(float(step)*BATCH_SIZE/len(train_list)), np.sum(l)/BATCH_SIZE, lr)
					del input_data, gt_data, cbcr_data

				saver.save(sess, "./checkpoints/pVDSR_const_clip_0.01_epoch_%03d.ckpt" % epoch ,global_step=global_step)



if __name__ == '__main__':
	main()
	
