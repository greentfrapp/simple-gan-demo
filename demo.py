"""
demo.py
A script for training a simple GAN and generates a video tracking the changes in the discriminator and generator output across training steps:
[1] Goodfellow, Ian, et al. "Generative adversarial nets." Advances in neural information processing systems. 2014.
http://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf
"""

from __future__ import print_function

import random
import tensorflow as tf
import numpy as np
import os
import datetime
import time
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy.interpolate import UnivariateSpline

class DenseNetwork(object):
	def __init__(self, nodes_per_layer, activations_per_layer, names_per_layer, network_name):
		self.name = network_name
		self.layers = []
		for layer_no, layer_name in enumerate(names_per_layer):
			self.layers.append({
				"name": layer_name,
				"nodes": nodes_per_layer[layer_no],
				"activation": activations_per_layer[layer_no]
				})
		return None
	def forwardprop(self, input_tensor, reuse_variables=False):
		if reuse_variables:
			tf.get_variable_scope().reuse_variables()
		with tf.name_scope(self.name):
			tensor = input_tensor
			for layer in self.layers:
				tensor = tf.layers.dense(
					inputs=tensor,
					units=layer["nodes"],
					activation=layer["activation"],
					kernel_initializer=tf.truncated_normal_initializer(.0,.01),
					name=layer["name"])
			return tensor

class GANDemo(object):
	def __init__(self, batch_size=1000):

		self.batch_size = batch_size
		self.learning_rate = 0.001

		# Networks
		self.generator = DenseNetwork(
			nodes_per_layer=[128, 128, 1],
			activations_per_layer=[tf.nn.relu, tf.nn.relu, None],
			names_per_layer=["generator_dense_1", "generator_dense_2", "generator_output"],
			network_name="Generator")
		self.discriminator = DenseNetwork(
			nodes_per_layer=[128, 128, 1],
			activations_per_layer=[tf.nn.relu, tf.nn.relu, None],
			names_per_layer=["discriminator_dense_1", "discriminator_dense_2", "discriminator_output"],
			network_name="Discriminator")

		# Placeholders
		self.input = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, 1], name='noise_input')
		self.real_prior = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, 1], name='real_prior')

		# Outputs from forwardprop-ing networks 
		with tf.variable_scope(tf.get_variable_scope()):
			self.fake_prior = self.generator.forwardprop(self.input)
			self.score_real_prior = self.discriminator.forwardprop(self.real_prior)
			self.score_fake_prior = self.discriminator.forwardprop(self.fake_prior, reuse_variables=True)

		self.output_scores = tf.sigmoid(self.score_real_prior)
		
		# Loss functions
		# For discriminator, 
		# 	label should be 1.0 if sample is from real prior, 0.0 if sample is from fake prior
		discriminator_loss_real_prior = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.score_real_prior), logits=self.score_real_prior))
		discriminator_loss_fake_prior = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(self.score_fake_prior), logits=self.score_fake_prior))
		self.discriminator_loss = discriminator_loss_real_prior + discriminator_loss_fake_prior
		# For generator,
		#	label should be 1.0 if sample is from fake prior, since it wants to fool the discriminator
		self.generator_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.score_fake_prior), logits=self.score_fake_prior))

		# Training functions
		all_variables = tf.trainable_variables()
		self.discriminator_variables = [var for var in all_variables if 'discriminator' in var.name]
		self.discriminator_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.discriminator_loss, var_list=self.discriminator_variables)
		self.generator_variables = [var for var in all_variables if 'generator' in var.name]
		self.generator_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.generator_loss, var_list=self.generator_variables)

		# Things to save in Tensorboard
		tf.summary.scalar(name="Discriminator Loss", tensor=self.discriminator_loss)
		tf.summary.scalar(name="Generator Loss", tensor=self.generator_loss)
		tf.summary.histogram(name="Encoder Distribution", values=self.fake_prior)
		tf.summary.histogram(name="Real Distribution", values=self.real_prior)
		self.summary_op = tf.summary.merge_all()

		# Initialize Tensorflow session and variables
		init = tf.global_variables_initializer()
		self.sess = tf.Session()
		self.sess.run(init)
		self.saver = tf.train.Saver()

		return None
		
	def sample_prior(self, size):
		# Real prior distribution, in this case just a Gaussian with shifted mean
		return np.random.randn(size, 1) + 3.0

	def sample_latent(self, size):
		# Latent vector for input to generator, in this case just a plain Gaussian
		return np.random.randn(size, 1)

	def create_checkpoint_folders(self, id_no):
		folder_name = "{}_gan_demo".format(id_no)
		subfolders = ["tensorboard", "saved_models", "log"]
		paths = ()
		for subfolder in subfolders:
			path = os.path.join(self.results_path, folder_name, subfolder)
			tf.gfile.MakeDirs(path)
			paths += (path,)
		return paths

	def get_loss(self, batch_x, z_real_dist):
		d_loss, g_loss, summary = self.sess.run([self.discriminator_loss, self.generator_loss, self.summary_op], feed_dict={self.input:batch_x, self.real_prior:z_real_dist})
		return (d_loss, g_loss, summary)

	def print_log(self, epoch, d_loss, g_loss):
		entry = "{}: Epoch #{}\n\tDiscriminator Loss - {}\n\tGenerator Loss - {}".format(datetime.datetime.now(), epoch, d_loss, g_loss)
		print(entry)
		with open(self.log_path + '/log.txt', 'a') as log:
			log.write(entry)

	def train(self, n_epochs=50):

		id_no = time.strftime('%Y%m%d_%H%M%S', datetime.datetime.now().timetuple())

		# Create results_folder
		self.results_path = 'results'
		tf.gfile.MakeDirs(self.results_path)

		self.n_epochs = n_epochs

		self.step = 0
		self.tensorboard_path, self.saved_model_path, self.log_path = self.create_checkpoint_folders(id_no)
		self.writer = tf.summary.FileWriter(logdir=self.tensorboard_path, graph=self.sess.graph)

		# Plotting parameters
		FFMpegWriter = animation.writers['ffmpeg']
		writer = FFMpegWriter(fps=5)
		fig = plt.figure()
		g_dist, = plt.plot([], [], label="Generator Output Distribution")
		prior_dist, = plt.plot([], [], label="Real Prior Distribution")
		d_dist, = plt.plot([], [], label="Discriminator Score")
		plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0., fontsize=10)
		min_x = -2
		max_x = 8
		plt.xlim(min_x, max_x)
		plt.ylim(-0.1, 1.1)

		video_path = "{}_gan_demo.mp4".format(id_no)

		with writer.saving(fig, video_path, 300):
			for epoch in range(1, self.n_epochs + 1):
				n_batches = 5

				for batch in range(n_batches):

					# Sample latent space and real prior
					batch_x = self.sample_latent(self.batch_size)
					z_real_dist = self.sample_prior(self.batch_size)

					# Get outputs from generator and discriminator
					fake_prior = self.sess.run(self.fake_prior, feed_dict={self.input: batch_x})
					scores = self.sess.run(self.output_scores, feed_dict={self.real_prior: np.expand_dims(np.arange(min_x, max_x, 0.01001), axis=1)})

					# Plot distributions of outputs for video
					bins = np.linspace(min_x, max_x, 50)
					
					fake_prior_y, fake_prior_binedges = np.histogram(fake_prior, bins=bins)
					fake_prior_bincenters = 0.5 * (fake_prior_binedges[1:] + fake_prior_binedges[:-1])
					fake_prior_y = fake_prior_y / float(max(fake_prior_y))
					
					prior_y, prior_binedges = np.histogram(z_real_dist, bins=bins)
					prior_bincenters = 0.5 * (prior_binedges[1:] + prior_binedges[:-1])
					prior_y = prior_y / float(max(prior_y))
					
					g_dist.set_data(bins[1:], fake_prior_y)
					prior_dist.set_data(bins[1:], prior_y)
					d_dist.set_data(np.arange(min_x, max_x, 0.01001), scores)
					
					writer.grab_frame()
					
					# Optimizer ops
					self.sess.run(self.discriminator_optimizer, feed_dict={self.input: batch_x, self.real_prior: z_real_dist})
					self.sess.run(self.generator_optimizer, feed_dict={self.input: batch_x})

					self.step += 1

				# Print log, write to log.txt, update Tensorboard and save model every epoch
				d_loss, g_loss, summary = self.get_loss(batch_x, z_real_dist)
				self.print_log(epoch, d_loss, g_loss)
				self.writer.add_summary(summary, global_step=self.step)
				self.saver.save(self.sess, save_path=self.saved_model_path, global_step=self.step)

		print("Model Trained!")
		print("Tensorboard Path: {}".format(self.tensorboard_path))
		print("Log Path: {}".format(self.log_path + '/log.txt'))
		print("Saved Model Path: {}".format(self.saved_model_path))
		print("Video Path: {}".format(video_path))
		return None

def main():
	gan = GANDemo()
	gan.train()

if __name__ == "__main__":
	main()






