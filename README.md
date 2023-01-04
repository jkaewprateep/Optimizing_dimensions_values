# Optimizing_dimensions_values
For Study Optimizing Dimensions values 3D, working with 3D objects its required dimensions to estimate of object shape and its neightbours. 

## Loss function optimizing values ##

We consider of both sigma values and minimum size scalable, optimizer adjusting of input dimensions value by sigma value called differentate equation as its nature optimize one dimension create feedback to next dimension with in the same equation. ``` Z = tf.nn.l2_loss( ( scale - sigma ) + ( scale - min_size ) , name="loss") ```

#### Sections run optimize #### 

Simply equation to optimizing input values with our custom optimizers class or equation ``` Z = tf.nn.l2_loss( ( scale - sigma ) + ( scale - min_size ) , name="loss") ``` , there are many loss functions for estimates of the new target approaches on the same pane but for 3D dimensions equation, they required the optimum way of optimizing and power of calculation savings when optimizing all dimensions is like you turn the scientists globe to find the location you know the answer then you can located both X and Y coordinates. 

```
with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
	
    for i in range(1000):
        global_step = global_step + 1

        train_loss, temp = sess.run([loss, training_op], feed_dict={scale:X, sigma:Y, min_size:Z})
        history.append(train_loss)
        history_Y.append( history[0] - train_loss )
		
        scale_value = scale.eval()
        sigma_value = sigma.eval()
        min_size_value = min_size.eval()

sess.close()
```

#### Segment Optimiztion TF 1.X #### 

```
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
```

#### Segment Optimiztion TF 2.X #### 

```
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
```

## Files and Directory ##

| File Name | Description |
--- | --- |
| sample.py | sample codes|
| Figure_2.png | Training Loss values |
| Figure_3.png | Target Shape |
| Figure_4.png | Optimizing Shape |
| Figure_5.png | Optimizing Shape |
| README.md | readme file |


## Results ##

![Loss values](https://github.com/jkaewprateep/Optimizing_dimensions_values/blob/main/Figure_2.png "Loss values")

![Shape](https://github.com/jkaewprateep/Optimizing_dimensions_values/blob/main/Figure_3.png "Shape")

![Optimizing Shape](https://github.com/jkaewprateep/Optimizing_dimensions_values/blob/main/Figure_4.png "Optimizing Shape")

![Series Prediction](https://github.com/jkaewprateep/Optimizing_dimensions_values/blob/main/Figure_5.png "Series Prediction")
