import tensorflow as tf

mobilenet = tf.keras.applications.MobileNetV2(input_shape = (224,224,3),
                                                                include_top=False, 
                                                                weights='imagenet')

 
average_pooling = tf.keras.layers.GlobalAveragePooling2D()(mobilenet.output)
new_output = tf.keras.layers.Dense(2, activation = 'softmax')(average_pooling)

model = tf.keras.Model(mobilenet.inputs, new_output)
model.summary()