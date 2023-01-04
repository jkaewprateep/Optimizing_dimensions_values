# Optimizing_dimensions_values
For Study Optimizing Dimensions values 3D, working with 3D objects its required dimensions to estimate of object shape and its neightbours. 

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

## Files and Directory ##

| File Name | Description |
--- | --- |
| sample.py | sample codes|
| Figure_2.png | Training Loss values |
| Figure_3.png | Target Shape |
| Figure_4.png | Optimizing Shape |
| README.md | readme file |


## Results ##

![Loss values](https://github.com/jkaewprateep/Optimizing_dimensions_values/blob/main/Figure_2.png "Loss values")

![Shape](https://github.com/jkaewprateep/Optimizing_dimensions_values/blob/main/Figure_3.png "Shape")

![Optimizing Shape](https://github.com/jkaewprateep/Optimizing_dimensions_values/blob/main/Figure_4.png "Optimizing Shape")
