#!/usr/bin/python
# -*- coding: utf-8 -*-
import sys
sys.path.append("..")
import tensorflow as tf
import numpy as np
from utils.util import *
from model.LSTMCRF import *

class Joint(LSTMCRF):
	def __init__(self, args):
		super(Joint,self).__init__(args)
		if args.algorithm=='Joint':
			self.build()

	def build(self):
		self.mix()
		if self.args.mode in ['train','tune']:
			self.loss_layer()

	def mix(self):
		encode1=self.mix_lstm1('net1')
		encode2=self.mix_lstm1('net2')

		# this place is easy to mistake
		if self.args.mask_net1==1:
			encode1=encode1*0.0
		if self.args.gradient_stop_net1==1:
			encode1=tf.stop_gradient(encode1)

		if self.args.mask_net2==1:
			encode2=encode2*0.0
		if self.args.gradient_stop_net2==1:
			encode2=tf.stop_gradient(encode2)
		
		output=[]
		if self.args.gradient_stop_net1 * self.args.gradient_stop_net2:
			tf.summary.image('encode1',tf.expand_dims(tf.expand_dims(encode1,axis=-1)[0],dim=0))
			tf.summary.image('encode2',tf.expand_dims(tf.expand_dims(encode2,axis=-1)[0],dim=0))
			# tf.summary.image('encode3',tf.expand_dims(tf.expand_dims(encode3,axis=-1)[0],dim=0))
			# tf.summary.image('encode4',tf.expand_dims(tf.expand_dims(encode4,axis=-1)[0],dim=0))
			encode=tf.concat([encode1,encode2],axis=-1)
			
			encode=self.dense(encode,100,'dense3-1',linear=False,bias=True)*self.mask_x
			encode=tf.nn.dropout(encode,self.dropout_keep_prob)
			output=self.dense(encode,self.output_class_num,'dense3-2',linear=True,bias=True)*self.mask_x
		
		else:
			# output1 : 50
			output1=self.dense(encode1,self.output_class_num,'l-left',linear=True,bias=True)*self.mask_x
			output2=self.dense(encode2,self.output_class_num,'l-right',linear=True,bias=True)*self.mask_x
			# output3=self.dense(encode3,self.output_class_num,'l-spanish',linear=True,bias=True)*self.mask_x
			# output4=self.dense(encode4,self.output_class_num,'l-dutch',linear=True,bias=True)*self.mask_x
			
			if self.args.mask_net1==0:
				output=output1
			if self.args.mask_net2==0:
				output=output2

			# output=output1

			# elif self.mask3==0:
			# 	output=output3
			# elif self.mask4==0:
			# 	output=output4

		self.output=output
		with tf.variable_scope('crf'):
			dim_crf1=self.output_class_num
			crf_log_likelihood_lstm1,transition_params_lstm1,W_crf_lstm1,b_crf_lstm1= \
			self.crf_loss_builtin(
				output
				,self.y_tag_sparse
				,dim_crf1
				,test=False
				,max_length=self.max_length
				,batch_actual_length=self.batch_actual_length)

			loss=tf.reduce_mean(-1.0*crf_log_likelihood_lstm1)
			tf.summary.scalar('crf_loss',loss)
			self.loss=loss

			self.viterbi_sequence=self.crf_predict(output
				,W_crf_lstm1
				,b_crf_lstm1
				,transition_params_lstm1
				,dim_crf1)
		print('class-num...',self.output_class_num)


	# These sample blocks will be further organized, and this toolkit will provide more easy-to-use blocks.
	# You can found it in https://liftkkkk.github.io/FLEXNER/