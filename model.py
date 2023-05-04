from tensorflow import keras

def get_model(width, height, num_classes):
    model = keras.models.Sequential()
    # 卷积层1，32个3*3的卷积核，输出维度为(48,48,32)，因为是灰度图像所以是(width,height,1)的
    model.add(keras.layers.Conv2D(filters=32,kernel_size=3,
                                padding='same',activation='relu',
                                input_shape=(width,height,1)))
    # model.add(keras.layers.BatchNormalization())

    # 卷积层2，32个3*3的卷积核，输出维度为(48,48,32)。
    model.add(keras.layers.Conv2D(filters=32,kernel_size=3,
                                padding='same',activation='relu'))
    # model.add(keras.layers.BatchNormalization())

    # 池化层1，2*2的池化核，输出维度为(24,24,32)。
    model.add(keras.layers.MaxPool2D(pool_size=2))

    # 卷积层3，64个3*3的卷积核，输出维度为(24,24,64)。
    model.add(keras.layers.Conv2D(filters=64,kernel_size=3,
                                padding='same',activation='relu'))
    # model.add(keras.layers.BatchNormalization())

    # 卷积层4，64个3*3的卷积核，输出维度为(24,24,64)。
    model.add(keras.layers.Conv2D(filters=64,kernel_size=3,
                                padding='same',activation='relu'))
    # model.add(keras.layers.BatchNormalization())

    # 池化层2，2*2的池化核，输出维度为(12,12,64)。
    model.add(keras.layers.MaxPool2D(pool_size=2))

    # 卷积层5，128个3*3的卷积核，输出维度为(12,12,128)。
    model.add(keras.layers.Conv2D(filters=128,kernel_size=3,
                                padding='same',activation='relu'))
    # model.add(keras.layers.BatchNormalization())

    # 卷积层6，128个3*3的卷积核，输出维度为(12,12,128)。
    model.add(keras.layers.Conv2D(filters=128,kernel_size=3,
                                padding='same',activation='relu'))
    # model.add(keras.layers.BatchNormalization())

    # 池化层3，2*2的池化核，输出维度为(6,6,128)。
    model.add(keras.layers.MaxPool2D(pool_size=2))

    # 扁平化层，将三维的矩阵转换为一维向量，输出维度为(66128)。
    model.add(keras.layers.Flatten())
    # model.add(keras.layers.AlphaDropout(0.5))

    # 全连接层1，128个神经元，输出维度为(128)。
    model.add(keras.layers.Dense(128,activation='relu'))

    # Dropout层，随机失活40%的神经元。
    model.add(keras.layers.Dropout(0.4))

    # 全连接层2，num_classes个神经元，输出维度为(num_classes)。
    model.add(keras.layers.Dense(num_classes,activation='softmax'))
    # adam=tf.optimizers.Adam(lr=0.01)
    # callbacks=[keras.callbacks.EarlyStopping(patience=5,min_delta=1e-3)]

    # 编译模型，其中使用adam优化器、交叉熵损失函数和精确度作为评估指标。
    model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'],
                )
    model.summary()
    return model