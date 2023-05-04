import tensorflow.keras as keras
def get_data_generators(train_folder, val_folder, test_folder, height, width, batch_size):
    # 进行数据增强
    # rescale：缩放因子，将像素值缩放到[0,1]范围内
    # rotation_range：随机旋转角度，范围为0~40度；
    # width_shift_range、height_shift_range：水平和垂直方向上的随机平移比例，范围为0~0.2；
    # shear_range：剪切强度，范围为0~0.2；
    # zoom_range：随机缩放比例，范围为0~0.2；
    # horizontal_flip：是否进行随机水平翻转；
    # fill_mode：填充方式，‘nearest’表示用最近邻插值填充。
    train_datagen=keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

    # 使用train_datagen.flow_from_directory()函数生成训练集、验证集和测试集的数据生成器。自动按照子文件夹名字将图像分类
    # train_folder、val_folder和test_folder分别为训练集、验证集和测试集的路径；
    # height和width为图像的高和宽；
    # batch_size为每批次的样本数；
    # seed为随机数种子，用于随机生成图像变换；
    # shuffle为是否打乱数据集顺序；
    # class_mode为分类模式，'categorical'表示多类别分类问题。
    # color_mode='grayscale'生成灰度图像。
    train_generator=train_datagen.flow_from_directory(
        train_folder,
        target_size=(height,width),
        batch_size=batch_size,
        seed=7,
        shuffle=True,
        #shuffle=False,
        color_mode='grayscale',
        class_mode='categorical')

    valid_datagen=keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255)
    valid_generator=train_datagen.flow_from_directory(
        val_folder,
        target_size=(height,width),
        batch_size=batch_size,
        seed=7,
        shuffle=False,
        color_mode='grayscale',
        class_mode='categorical')

    test_datagen=keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255)
    test_generator=test_datagen.flow_from_directory(
        test_folder,
        target_size=(height,width),
        batch_size=batch_size,
        seed=7,
        shuffle=False,
        color_mode='grayscale',
        class_mode='categorical')
    
    return train_generator, valid_generator, test_generator