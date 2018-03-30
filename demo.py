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

class GAN(object):
	def __init__(self, batch_size=1000):

		self.batch_size = batch_size
		self.learning_rate = 0.001

		# Networks
		self.generator = DenseNetwork(
			nodes_per_layer=[1000, 1000, 1],
			activations_per_layer=[tf.nn.relu, tf.nn.relu, None],
			names_per_layer=["generator_dense_1", "generator_dense_2", "generator_output"],
			network_name="Generator")
		self.discriminator = DenseNetwork(
			nodes_per_layer=[1000, 1000, 1],
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
		tf.summary.histogram(name="Encoder Distribution", values=self.counterfeit)
		tf.summary.histogram(name="Real Distribution", values=self.real_prior)
		self.summary_op = tf.summary.merge_all()

		# Initialize Tensorflow session and variables
		init = tf.global_variables_initializer()
		self.sess = tf.Session()
		self.sess.run(init)
		self.saver = tf.train.Saver()

		return None
		
	def sample_prior(self, size):
		# Real prior distribution, in this case just a Gaussian
		return np.random.randn(size, 1)

	def create_checkpoint_folders(self, batch_size, n_epochs):
		folder_name = "/{0}_{1}_{2}_GAN".format(
			datetime.datetime.now(),
			batch_size,
			n_epochs).replace(':', '-')
		tensorboard_path = self.results_path + folder_name + '/tensorboard'
		saved_model_path = self.results_path + folder_name + '/saved_models/'
		log_path = self.results_path + folder_name + '/log'
		if not os.path.exists(self.results_path + folder_name):
			os.mkdir(self.results_path + folder_name)
			os.mkdir(tensorboard_path)
			os.mkdir(saved_model_path)
			os.mkdir(log_path)
		return tensorboard_path, saved_model_path, log_path

	def get_loss(self, batch_x, z_real_dist):
		d_loss, g_loss, summary = self.sess.run([self.discriminator_loss, self.generator_loss, self.summary_op], feed_dict={self.input:batch_x, self.real_prior:z_real_dist})
		return (d_loss, g_loss, summary)

	def train(self, n_epochs=500):

		# Create results_folder
		self.results_path = 'results'
		if not os.path.exists(self.results_path):
			os.mkdir(self.results_path)

		self.n_epochs = n_epochs

		self.step = 0
		self.tensorboard_path, self.saved_model_path, self.log_path = self.create_checkpoint_folders(self.batch_size, self.n_epochs)
		self.writer = tf.summary.FileWriter(logdir=self.tensorboard_path, graph=self.sess.graph)

		# plotting stuff
		FFMpegWriter = animation.writers['ffmpeg']
		writer = FFMpegWriter(fps=5)
		fig = plt.figure()
		arti_dist, = plt.plot([], [], label="Artificial Distribution")
		prior_dist, = plt.plot([], [], label="Prior Distribution")
		score_dist, = plt.plot([], [], label="Discriminator Score")
		plt.legend(loc=2, frameon=False, fontsize=12)
		plt.xlim(-5, 5)
		plt.ylim(-0.1, 1.1)

		with writer.saving(fig, "writer_test.mp4", 300):
			for epoch in range(1, self.n_epochs + 1):
				n_batches = 1
				print("------------------Epoch {}/{}------------------".format(epoch, self.n_epochs))

				for batch in range(1, n_batches + 1):
					# batch_x = np.random.randn(self.batch_size, 1) * 2.0 + 1.0
					batch_x = np.random.rand(self.batch_size, 1)
					z_real_dist = self.sample_prior(self.batch_size)

					#plot_x = np.expand_dims(np.arange(-5, 5, 0.05), axis=1)
					counterfeit = self.sess.run(self.counterfeit, feed_dict={self.input: batch_x})
					scores = self.sess.run(self.output_scores, feed_dict={self.real_prior: np.expand_dims(np.arange(-5, 5, 0.01001), axis=1)})

					bins = np.linspace(-5, 5, 50)
					counterfeit_y, counterfeit_binedges = np.histogram(counterfeit, bins=bins)
					counterfeit_bincenters = 0.5 * (counterfeit_binedges[1:] + counterfeit_binedges[:-1])
					counterfeit_y = counterfeit_y / float(max(counterfeit_y))
					# counterfeit_spline = UnivariateSpline(counterfeit_bincenters, counterfeit_y, s=100)
					# arti_dist.set_data(counterfeit_bincenters, counterfeit_y)
					arti_dist.set_data(bins[1:], counterfeit_y)
					# arti_dist.set_data(counterfeit_bincenters, counterfeit_spline(counterfeit_bincenters))
					prior_y, prior_binedges = np.histogram(z_real_dist, bins=bins)
					prior_bincenters = 0.5 * (prior_binedges[1:] + prior_binedges[:-1])
					prior_y = prior_y / float(max(prior_y))
					# prior_spline = UnivariateSpline(prior_bincenters, prior_y, s=100)
					# prior_dist.set_data(prior_bincenters, prior_y)
					prior_dist.set_data(bins[1:], prior_y)
					# prior_dist.set_data(prior_bincenters, prior_spline(prior_bincenters))
					score_dist.set_data(np.arange(-5, 5, 0.01001), scores)
					writer.grab_frame()
					
					self.sess.run(self.discriminator_optimizer, feed_dict={self.input: batch_x, self.real_prior: z_real_dist})
					self.sess.run(self.generator_optimizer, feed_dict={self.input: batch_x})

					# Print log and write to log.txt every 50 batches
					if batch % 50 == 0:
						a_loss, d_loss, e_loss, summary = self.get_loss(batch_x, z_real_dist)
						self.writer.add_summary(summary, global_step=self.step)
						print("Epoch: {}, iteration: {}".format(epoch, batch))
						print("Discriminator Loss: {}".format(d_loss))
						print("Generator Loss: {}".format(g_loss))
						with open(self.log_path + '/log.txt', 'a') as log:
							log.write("Epoch: {}, iteration: {}\n".format(epoch, batch))
							log.write("Discriminator Loss: {}\n".format(d_loss))
							log.write("Generator Loss: {}\n".format(g_loss))

					self.step += 1

					# if epoch % 100 == 0:
					# 	quit()

				self.saver.save(self.sess, save_path=self.saved_model_path, global_step=self.step)

		print("Model Trained!")
		print("Tensorboard Path: {}".format(self.tensorboard_path))
		print("Log Path: {}".format(self.log_path + '/log.txt'))
		print("Saved Model Path: {}".format(self.saved_model_path))
		return None

def main():
	gan = GAN()
	gan.train()

if __name__ == "__main__":
	main()






