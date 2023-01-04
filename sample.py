
# https://stackoverflow.com/questions/72385854/get-layers-weights-inside-the-call-function-in-keras
# Get Layer's weights inside the call function in Keras
# If I have an optimization algorithm that has three parameters i.e., scale, sigma, and min_size. And I need to optimize these parameters with the parameters of a deep neural net. The problem I face is that I cant access the three weights I define for a custom Keras Layer inside the call function. As the code bellow shows, I want to access the weights I define in the call function such that I apply the optimization algorithm with the new learned parameters. Kindly, if anyone can help?

import os
from os.path import exists

import tensorflow as tf
import tensorflow_io as tfio

import matplotlib.pyplot as plt
import numpy as np

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
: Variables
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
learning_rate = 0.001
global_step = 0
tf.compat.v1.disable_eager_execution()

history = [ ] 
history_Y = [ ]

scale = 1.0
sigma = 1.0
min_size = 1.0

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
: Class / Function
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
class MyDenseLayer(tf.keras.layers.Layer):
	def __init__(self, num_outputs, num_add):
		super(MyDenseLayer, self).__init__()
		self.num_outputs = num_outputs
		self.num_add = num_add
		
	def build(self, input_shape):
		self.kernel = self.add_weight("kernel",
		shape=[int(input_shape[-1]),
		self.num_outputs])

	def call(self, inputs):
		temp = tf.add( inputs, self.num_add )
		temp = tf.matmul(temp, self.kernel)
		return temp

#####################################################################################################

class SeqmentationOptimization(tf.keras.layers.Layer):
    def __init__(self):
        super(SeqmentationOptimization, self).__init__()
        scale_init = tf.keras.initializers.RandomUniform(minval=10, maxval=1000, seed=None)
        sigma_init = tf.keras.initializers.RandomUniform(minval=0.001, maxval=1, seed=None)
        min_size_init = tf.keras.initializers.RandomUniform(minval=10, maxval=1000, seed=None)

        self.scale = self.add_weight(shape=[1],
                                initializer = scale_init,
                                trainable=True)
        self.sigma = self.add_weight(shape=[1],
                                initializer = sigma_init,
                                trainable=True)
        
        self.min_size = self.add_weight(shape=[1],
                                initializer = min_size_init,
                                trainable=True)

    def call(self, inputs):
        objects = Segmentation(self.scale , self.sigma , self.min_size ).objects
        return 
		

class Segmentation( ):
	def __init__( self, scale , sigma , min_size ):
		
		print( 'start __init__: ' )
		self.scale = scale
		self.sigma = sigma
		self.min_size = min_size
		
		scale = tf.compat.v1.get_variable('scale', dtype = tf.float32, initializer = tf.random.normal((1, 10, 1)))
		sigma = tf.compat.v1.get_variable('sigma', dtype = tf.float32, initializer = tf.random.normal((1, 10, 1)))
		min_size = tf.compat.v1.get_variable('min_size', dtype = tf.float32, initializer = tf.random.normal((1, 10, 1)))
		
		Z = tf.nn.l2_loss( ( scale - sigma ) +( scale - min_size ) , name="loss")
		loss = tf.reduce_mean(input_tensor=tf.square(Z))
		
		optimizer = tf.compat.v1.train.ProximalAdagradOptimizer(
		learning_rate,
		initial_accumulator_value=0.1,
		l1_regularization_strength=0.2,
		l2_regularization_strength=0.1,
		use_locking=False,
		name='ProximalAdagrad'
		)
		training_op = optimizer.minimize(loss)
		
		self.loss = loss
		self.scale = scale
		self.sigma = sigma
		self.min_size = min_size
		self.training_op = training_op
		
		return 
	
	def create_loss_fn( self ):
		print( 'start create_loss_fn: ' )
		
		return self.loss, self.scale, self.sigma, self.min_size, self.training_op
	

X = np.reshape([ 500, -400, 400, -300, 300, -200, 200, -100, 100, 1 ], (1, 10, 1))
Y = np.reshape([ -400, 400, -300, 300, -200, 200, -100, 100, -50, 50 ], (1, 10, 1))
Z = np.reshape([ -100, 200, -300, 300, -400, 400, -500, 500, -50, 50 ], (1, 10, 1))


loss_segmentation = Segmentation( scale , sigma , min_size )
loss, scale, sigma, min_size, training_op = loss_segmentation.create_loss_fn( )

scale_value = []
sigma_value = []
min_size_value = []
	
with tf.compat.v1.Session() as sess:
	sess.run(tf.compat.v1.global_variables_initializer())
	
	for i in range(1000):
		global_step = global_step + 1
		
		# train_loss, temp = sess.run([loss, training_op], feed_dict={X_var:X, y_var:Y})
		train_loss, temp = sess.run([loss, training_op], feed_dict={scale:X, sigma:Y, min_size:Z})
		history.append(train_loss)
		history_Y.append( history[0] - train_loss )
		
		scale_value = scale.eval()
		sigma_value = sigma.eval()
		min_size_value = min_size.eval()
		
		print( 'steps: ' + str(i) + " scale: " + str( scale_value ) + " sigma: " + str( sigma_value ) + " min_size: " + str( min_size_value ) 

		
		)
		print( scale_value.shape )
		
sess.close()

plt.plot(np.asarray(history))
plt.plot(np.asarray(history_Y))
plt.show()
plt.close()


scale = np.reshape( scale_value, (scale_value.shape[1], ) )
X = np.reshape( X, (scale_value.shape[1], ) )

x_scales = np.arange(0, scale.shape[0])


plt.plot( x_scales, scale )
plt.plot( x_scales, X )
plt.show()
plt.close()

input('...')

