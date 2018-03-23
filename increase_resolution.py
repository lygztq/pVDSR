import numpy as np
from scipy import misc
from PIL import Image
import tensorflow as tf
from model import model


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--img")
args = parser.parse_args()
img_path = args.img

def im2double(img):
	return img.astype(np.float)/255

def double2im(img):
	return (img*255).astype(np.int)

def YCbCr2rgb(ycbcr_img):
	img = ycbcr_img.astype(np.int)
	rgb = np.zeros([img.shape[0],img.shape[1],img.shape[2]],dtype=np.float32)
	rgb[:,:,0] = ((img[:,:,0] - 16)*298.082 + (img[:,:,2]-128)*408.583)/256
	rgb[:,:,1] = ((img[:,:,0] - 16)*298.082 - (img[:,:,1]-128)*100.291 - (img[:,:,2]-128)*208.12)/256
	rgb[:,:,2] = ((img[:,:,0] - 16)*298.082 + (img[:,:,1]-128)*516.411)/256
	rgb = rgb.astype(np.int)
	return rgb
if __name__ == '__main__':
	ckpt_dir = './checkpoints'
	img = misc.imread(img_path, mode='YCbCr')
	img_Y ,img_Cb, img_Cr = img[:,:,0], img[:,:,1], img[:,:,2]
	img_Y = im2double(img_Y)

	with tf.Session() as sess:
		input_tensor = tf.placeholder(tf.float32, shape=(1, None, None, 1))
		shared_model = tf.make_template('shared_model', model)
		output_tensor, weights = shared_model(input_tensor)
		init = tf.global_variables_initializer()
		sess.run(init)
		ckpt = tf.train.get_checkpoint_state(ckpt_dir)
		saver = tf.train.Saver(weights)
		saver.restore(sess, tf.train.latest_checkpoint(ckpt_dir))
		output_Y = sess.run([output_tensor], feed_dict={input_tensor: np.resize(img_Y, (1, img_Y.shape[0], img_Y.shape[1], 1))})
		output_Y = np.resize(output_Y, (img_Y.shape[0], img_Y.shape[1]))
		out_path = img_path.split('.')[0] + 'out.' + img_path.split('.')[-1]
		output_Y = double2im(output_Y)
		img = img.astype(np.int)
		img[:,:,0] = output_Y
		rgb_img = YCbCr2rgb(img)
		misc.imsave(out_path,rgb_img)

