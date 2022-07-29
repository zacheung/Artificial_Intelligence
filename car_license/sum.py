"""
@author: Zach
@time:   2022/7/26 上午11:21
@E-mail: one.bud@foxmail.com
@IDE:    PyCharm
@File:   sum.py
@Intro:  识别车牌，分割车牌字符，利用模型进行预测
"""
import cv2
import time
import numpy as np
from PIL import Image
import tensorflow as tf

print(tf.__version__)  # 1.6.0
SIZE = 1280
WIDTH = 32
HEIGHT = 40
PROVINCES = ('京', '闽', '粤', '苏', '沪', '浙')
LETTERS = ("A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z")
LETTERS_DIGITS = ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "P",
                  "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z")
time_begin = time.time()


def find_car_num_brod():
    """识别出车牌并将车牌剪切保存"""
    # CascadeClassifier()是OpenCV中做人脸检测的时候的一个级联分类器。既可以使用Haar，也可以使用LBP特征。
    watch_cascade = cv2.CascadeClassifier('./cascade.xml')
    # 先读取图片
    image = cv2.imread('./images/car_image/su.jpg')  # (327, 496, 3)
    resize_h = 1000
    scale = image.shape[1] / float(image.shape[0])  # 496/327=1.5168195718654434
    # resize()对图片进行缩放，width和height可以自己任意指定，不论大小。
    image = cv2.resize(image, (int(scale * resize_h), resize_h))  # (1000, 1516, 3) 此处不是按比例缩放，宽被放大了
    image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  # (1000, 1516) cv2.cvtColor()进行色彩空间的转换
    # detectMultiScale可以检测出图片中所有的人脸，并将人脸用vector保存各个人脸的坐标、大小（用矩形表示）
    # scaleFactor--表示在前后两次相继的扫描中，搜索窗口的比例系数。默认为1.1即每次搜索窗口依次扩大10%；
    # minNeighbors--表示构成检测目标的相邻矩形的最小个数(默认为3个)；minSize和maxSize用来限制得到的目标区域的范围
    # 此处检测出车牌的左上角坐标和车牌的长宽
    watches = watch_cascade.detectMultiScale(image_gray, 1.2, minNeighbors=4, minSize=(36, 9), maxSize=(106 * 40, 59 * 40))  # (1, 4)

    print('检测到车牌数：', len(watches))
    if len(watches) == 0:
        return False
    for x, y, w, h in watches:
        print(x, y, w, h)
        # rectangle()在图片上画长方形，坐标原点是图片左上角，向右为x轴正方向，向下为y轴正方向
        # 参数表示依次为：图片，长方形框左上角坐标，长方形框右下角坐标，字体颜色，字体粗细
        after_rectangle = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 1)  # (1000, 1516, 3)
        # cv2.imshow('rectangle', after_rectangle)  # 展示画框后的图片
        # cv2.waitKey(0)

        cut_img = image[y + 5:y + h - 5, x + 8:x + w - 8]  # (136, 558, 3) 裁剪坐标为[y0:y1, x0:x1]
        cut_gray = cv2.cvtColor(cut_img, cv2.COLOR_RGB2GRAY)  # (136, 558) cv2.cvtColor()进行色彩空间的转换
        # cv2.imshow('rectangle', cut_gray)  # cv2.imshow()函数需要两个输入，一个是图像窗口的名字即title，一个是所展示图片的像素值矩阵
        # cv2.waitKey(5000)  # waitKey(k)在时间k(单位ms)内，等待用户按键触发，如果没有触发事件，则跳出等待。若k=0，则无限等待触发事件。
        cv2.imwrite('./images/num_for_car.jpg', cut_gray)  # 图片像素为558 × 136

        im = Image.open('./images/num_for_car.jpg')
        size = 720, 180
        mmm = im.resize(size, Image.ANTIALIAS)  # antialias：平滑；反走样；抗锯齿
        # quality参数：保存图像的质量，值的范围从1（最差）到95（最佳）
        # 默认值为75，使用中应尽量避免高于95的值，100会禁用部分JPEG压缩算法，并导致大文件图像质量几乎没有任何增益。
        mmm.save('./images/num_for_car.jpg', 'JPEG', quality=95)  # 重新保存图片像素为720 × 180，目的是为了提高图片质量
    return True


def find_end(start, white, black, arg, white_max, black_max, width):
    """查找字符结束位置下标"""
    end = start + 1
    for m in range(start + 1, width):
        # 当二值化成功arg为True时，将每一列黑色像素总和与黑色像素最多列*0.95比较，大于则认为字符结束
        if (black[m] if arg else white[m]) > (0.95 * black_max if arg else 0.95 * white_max):  # 0.95这个参数请多调整，对应下面的0.05
            end = m
            break
    return end


def cut_car_num_for_chart():
    """读取车牌图片分割字符并保存"""
    # 1. 读取图像，并把图像转换为灰度图像显示
    img = cv2.imread('./images/num_for_car.jpg')  # 读取图片
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 灰度化
    # cv2.imshow('gray', img_gray)  # 显示图片
    # cv2.waitKey(0)

    # 2. 将灰度图像二值化，设定阈值是100，转换为黑底白字
    # ksize – 高斯核大小。 ksize.width 并且 ksize.height 可以有所不同，但它们都必须是正数和奇数
    # sigmaX – X方向上的高斯核标准偏差
    blur = cv2.GaussianBlur(img_gray, (5, 5), 0)  # 高斯除噪
    # src是灰度图像，thresh是起始阈值，maxval是最大值，type是定义如何处理数据与阈值的关系
    ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # ret3=153.0 二值化处理 cv2.THRESH_OTSU使用最小二乘法处理像素点
    cv2.imwrite('./images/wb_img.jpg', th3)

    # 3. 分割字符并保存
    white, black = [], []  # 记录每一列的白色和黑色像素总和
    height, width = th3.shape[0], th3.shape[1]  # 180 720
    white_max, black_max = 0, 0
    for i in range(width):  # 计算每一列的黑白像素总和
        s, t = 0, 0  # s这一列白色总数，t这一列黑色总数
        for j in range(height):
            if th3[j][i] == 255:
                s += 1
            if th3[j][i] == 0:
                t += 1
        white_max = max(white_max, s)
        black_max = max(black_max, t)
        white.append(s)
        black.append(t)
    print(len(white), len(black))  # 720 720
    print('black_max:' + str(black_max) + ' white_max:' + str(white_max))  # 黑色像素最多的一列像素数，白色像素最多的一列像素数
    arg = False  # False表示白底黑字；True表示黑底白字。防止上面二值化失败
    if black_max > white_max:
        arg = True

    n = 0
    split_name = 1  # 分割图片的名字
    while n < width - 2:  # 720-2=718 width - 2
        # 当二值化成功arg为True时，将每一列白色像素总和与白色像素最多列*0.05比较，大于则认为有字符
        if (white[n] if arg else black[n]) > (0.05 * white_max if arg else 0.05 * black_max):  # 0.05这个参数请多调整，对应上面的0.95
            # print('-' * 10, n, white[n], black[n], 0.05 * white_max, 0.05 * black_max)
            start = n  # 字符开始位置下标
            end = find_end(start, white, black, arg, white_max, black_max, width)  # 查找字符结束位置下标
            n = end  # 改变循环n的下一次开始位置

            if end - start > 5:  # 移除车牌两侧的白边
                print('split_name:' + str(split_name), ' end-start=' + str(end - start))  # 输出切割的字符宽度
                cj = th3[1:height, start:end]
                cv2.imwrite('./images/img_cut_not_3240/' + str(split_name) + '.jpg', cj)  # 保存分割出来的单个字符

                im = Image.open('./images/img_cut_not_3240/' + str(split_name) + '.jpg')
                size = 32, 40
                mmm = im.resize(size, Image.ANTIALIAS)  # 进行大小缩放
                mmm.save('./images/img_cut/' + str(split_name) + '.bmp', quality=95)  # 重新保存
                split_name += 1

        else:
            # print(n, white[n], black[n], 0.05 * white_max, 0.05 * black_max)
            pass

        n += 1


# 定义卷积函数
def conv_layer(inputs, W, b, conv_strides, kernel_size, pool_strides, padding):
    L1_conv = tf.nn.conv2d(inputs, W, strides=conv_strides, padding=padding)  # 给定4维的输入张量和滤波器张量来进行2维的卷积计算
    L1_relu = tf.nn.relu(L1_conv + b)  # ReLu非线性变换
    return tf.nn.max_pool(L1_relu, ksize=kernel_size, strides=pool_strides, padding='SAME')  # 最大池化


# 定义全连接层函数
def full_connect(inputs, W, b):
    return tf.nn.relu(tf.matmul(inputs, W) + b)


def province_test():
    """加载已经训练好的模型并对分割好的车牌省份简称进行预测"""
    province_graph = tf.Graph()  # 生成新的计算图
    with province_graph.as_default():  # 将province_graph设置为默认图，并返回一个上下文管理器
        with tf.Session(graph=province_graph) as sess_p:
            # 定义输入节点，对应于图片像素值矩阵集合和图片标签（即所代表的数字）
            x = tf.placeholder(tf.float32, shape=[None, SIZE])
            x_image = tf.reshape(x, [-1, WIDTH, HEIGHT, 1])
            saver_p = tf.train.import_meta_graph('./train_saver/province/model.ckpt.meta')
            model_file = tf.train.latest_checkpoint('./train_saver/province')
            saver_p.restore(sess_p, model_file)

            # 第一个卷积层
            W_conv1 = sess_p.graph.get_tensor_by_name('W_conv1:0')
            b_conv1 = sess_p.graph.get_tensor_by_name('b_conv1:0')
            conv_strides = [1, 1, 1, 1]
            kernel_size = [1, 2, 2, 1]
            pool_strides = [1, 2, 2, 1]
            L1_pool = conv_layer(x_image, W_conv1, b_conv1, conv_strides, kernel_size, pool_strides,
                                 padding='SAME')  # Tensor("MaxPool_2:0", shape=(?, 16, 20, 16), dtype=float32) (?, 16, 20, 16)

            # 第二个卷积层
            W_conv2 = sess_p.graph.get_tensor_by_name('W_conv2:0')  # Tensor("W_conv2:0", shape=(5, 5, 16, 32), dtype=float32_ref)
            b_conv2 = sess_p.graph.get_tensor_by_name('b_conv2:0')  # Tensor("b_conv2:0", shape=(32,), dtype=float32_ref)
            conv_strides = [1, 1, 1, 1]
            kernel_size = [1, 1, 1, 1]
            pool_strides = [1, 1, 1, 1]
            L2_pool = conv_layer(L1_pool, W_conv2, b_conv2, conv_strides, kernel_size, pool_strides,
                                 padding='SAME')  # Tensor("MaxPool_3:0", shape=(?, 16, 20, 32), dtype=float32)

            # 全连接层
            W_fc1 = sess_p.graph.get_tensor_by_name('W_fc1:0')
            b_fc1 = sess_p.graph.get_tensor_by_name('b_fc1:0')
            h_pool2_flat = tf.reshape(L2_pool, [-1, 16 * 20 * 32])  # (?, 10240)
            h_fc1 = full_connect(h_pool2_flat, W_fc1, b_fc1)  # (?, 10240)

            # dropout
            keep_prob = tf.placeholder(tf.float32)
            h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

            # readout层
            W_fc2 = sess_p.graph.get_tensor_by_name('W_fc2:0')
            b_fc2 = sess_p.graph.get_tensor_by_name('b_fc2:0')

            # 定义优化器和训练op
            conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

            for n in range(1, 2):
                path = './images/img_cut/%s.bmp' % n
                img = Image.open(path)
                width, height = img.size[0], img.size[1]
                img_data = [[0] * SIZE for i in range(1)]  # len(img_data[0])==1280
                for h in range(0, height):
                    for w in range(0, width):
                        if img.getpixel((w, h)) < 190:
                            img_data[0][w + h * width] = 1
                        else:
                            img_data[0][w + h * width] = 0
                result = sess_p.run(conv, feed_dict={x: np.array(img_data), keep_prob: 1.0})  # result[0].shape=(6,)
                sorted_result = np.sort(result)  # 对预测结果概率进行排序
                max1, max2, max3 = sorted_result[0][-1], sorted_result[0][-2], sorted_result[0][-3]  # 选出前三个高的概率值
                max1_index, max2_index, max3_index = 0, 0, 0
                for j in range(len(PROVINCES)):  # 查找前三个概率值对应的下标
                    if result[0][j] == max1:
                        max1_index = j
                    if result[0][j] == max2:
                        max2_index = j
                    if result[0][j] == max3:
                        max3_index = j
                print('概率：[%s %0.4f%%]  [%s %0.4f%%]    [%s %0.4f%%]' % (
                    PROVINCES[max1_index], max1 * 100, PROVINCES[max2_index], max2 * 100, PROVINCES[max3_index], max3 * 100))
            print('省份简称是：%s' % PROVINCES[max1_index])
            return PROVINCES[max1_index]


def letter_test():
    """加载已经训练好的模型并对分割好车牌的地级市字母进行预测"""
    letter_graph = tf.Graph()
    with letter_graph.as_default():
        with tf.Session(graph=letter_graph) as sess:
            # 定义输入节点，对应于图片像素值矩阵集合和图片标签（即所代表的数字）
            x = tf.placeholder(tf.float32, shape=[None, SIZE])
            x_image = tf.reshape(x, [-1, WIDTH, HEIGHT, 1])
            saver = tf.train.import_meta_graph('./train_saver/letters/model.ckpt.meta')
            model_file = tf.train.latest_checkpoint('./train_saver/letters')
            saver.restore(sess, model_file)

            # 第一个卷积层
            W_conv1 = sess.graph.get_tensor_by_name('W_conv1:0')
            b_conv1 = sess.graph.get_tensor_by_name('b_conv1:0')
            conv_strides = [1, 1, 1, 1]
            kernel_size = [1, 2, 2, 1]
            pool_strides = [1, 2, 2, 1]
            L1_pool = conv_layer(x_image, W_conv1, b_conv1, conv_strides, kernel_size, pool_strides, padding='SAME')

            # 第二个卷积层
            W_conv2 = sess.graph.get_tensor_by_name('W_conv2:0')
            b_conv2 = sess.graph.get_tensor_by_name('b_conv2:0')
            conv_strides = [1, 1, 1, 1]
            kernel_size = [1, 1, 1, 1]
            pool_strides = [1, 1, 1, 1]
            L2_pool = conv_layer(L1_pool, W_conv2, b_conv2, conv_strides, kernel_size, pool_strides, padding='SAME')

            # 全连接层
            W_fc1 = sess.graph.get_tensor_by_name('W_fc1:0')
            b_fc1 = sess.graph.get_tensor_by_name('b_fc1:0')
            h_pool2_flat = tf.reshape(L2_pool, [-1, 16 * 20 * 32])
            h_fc1 = full_connect(h_pool2_flat, W_fc1, b_fc1)

            # dropout
            keep_prob = tf.placeholder(tf.float32)
            h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

            # readout层
            W_fc2 = sess.graph.get_tensor_by_name('W_fc2:0')
            b_fc2 = sess.graph.get_tensor_by_name('b_fc2:0')

            # 定义优化器和训练op
            conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

            for n in range(2, 3):
                path = './images/img_cut/%s.bmp' % n
                img = Image.open(path)
                width, height = img.size[0], img.size[1]
                img_data = [[0] * SIZE for i in range(1)]  # len(img_data[0])==1280

                for h in range(0, height):
                    for w in range(0, width):
                        if img.getpixel((w, h)) < 190:
                            img_data[0][w + h * width] = 1
                        else:
                            img_data[0][w + h * width] = 0
                result = sess.run(conv, feed_dict={x: np.array(img_data), keep_prob: 1.0})  # (1, 26)
                sorted_result = np.sort(result)  # 对预测结果概率进行排序
                max1, max2, max3 = sorted_result[0][-1], sorted_result[0][-2], sorted_result[0][-3]  # 选出前三个概率值
                max1_index, max2_index, max3_index = 0, 0, 0
                for j in range(len(LETTERS)):
                    if result[0][j] == max1:
                        max1_index = j
                    if result[0][j] == max2:
                        max2_index = j
                    if result[0][j] == max3:
                        max3_index = j

                print('概率：[%s %0.2f%%]  [%s %0.2f%%]    [%s %0.2f%%]' % (
                    LETTERS[max1_index], max1 * 100, LETTERS[max2_index], max2 * 100, LETTERS[max3_index], max3 * 100))
            print('城市代号是：[%s]' % LETTERS[max1_index])
            return LETTERS[max1_index]


def last_5_num_test():
    license_num = ''
    last_5_num_graph = tf.Graph()
    with last_5_num_graph.as_default():
        with tf.Session(graph=last_5_num_graph) as sess:
            # 定义输入节点，对应于图片像素值集合和图片标签（即所代表的数字）
            x = tf.placeholder(tf.float32, shape=[None, SIZE])
            x_image = tf.reshape(x, [-1, WIDTH, HEIGHT, 1])
            saver = tf.train.import_meta_graph('./train_saver/letters_digits/model.ckpt.meta')
            model_file = tf.train.latest_checkpoint('./train_saver/letters_digits')
            saver.restore(sess, model_file)

            # 第一个卷积层
            W_conv1 = sess.graph.get_tensor_by_name('W_conv1:0')
            b_conv1 = sess.graph.get_tensor_by_name('b_conv1:0')
            conv_strides = [1, 1, 1, 1]
            kernel_size = [1, 2, 2, 1]
            pool_strides = [1, 2, 2, 1]
            L1_pool = conv_layer(x_image, W_conv1, b_conv1, conv_strides, kernel_size, pool_strides, padding='SAME')

            # 第二个卷积层
            W_conv2 = sess.graph.get_tensor_by_name('W_conv2:0')
            b_conv2 = sess.graph.get_tensor_by_name('b_conv2:0')
            conv_strides = [1, 1, 1, 1]
            kernel_size = [1, 1, 1, 1]
            pool_strides = [1, 1, 1, 1]
            L2_pool = conv_layer(L1_pool, W_conv2, b_conv2, conv_strides, kernel_size, pool_strides, padding='SAME')

            # 全连接层
            W_fc1 = sess.graph.get_tensor_by_name('W_fc1:0')
            b_fc1 = sess.graph.get_tensor_by_name('b_fc1:0')
            h_pool2_flat = tf.reshape(L2_pool, [-1, 16 * 20 * 32])
            h_fc1 = full_connect(h_pool2_flat, W_fc1, b_fc1)

            # dropout
            keep_prob = tf.placeholder(tf.float32)
            h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

            # readout层
            W_fc2 = sess.graph.get_tensor_by_name('W_fc2:0')
            b_fc2 = sess.graph.get_tensor_by_name('b_fc2:0')

            # 定义优化器和训练op
            conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
            for n in range(4, 9):
                path = './images/img_cut/%s.bmp' % n
                img = Image.open(path)
                width, height = img.size[0], img.size[1]
                img_data = [[0] * SIZE for i in range(1)]  # len(img_data[0])=1280
                for h in range(0, height):
                    for w in range(0, width):
                        if img.getpixel((w, h)) < 190:
                            img_data[0][w + h * width] = 1
                        else:
                            img_data[0][w + h * width] = 0

                result = sess.run(conv, feed_dict={x: np.array(img_data), keep_prob: 1.0})
                sorted_result = np.sort(result)
                max1, max2, max3 = sorted_result[0][-1], sorted_result[0][-2], sorted_result[0][-3]  # 选出前三个高的概率值
                max1_index, max2_index, max3_index = 0, 0, 0
                for j in range(len(LETTERS_DIGITS)):  # 查找前三个概率值对应的下标
                    if result[0][j] == max1:
                        max1_index = j
                    if result[0][j] == max2:
                        max2_index = j
                    if result[0][j] == max3:
                        max3_index = j

                license_num = license_num + LETTERS_DIGITS[max1_index]
                print('概率：[%s %0.2f%%]  [%s %0.2f%%]    [%s %0.2f%%]' % (
                    LETTERS_DIGITS[max1_index], max1 * 100, LETTERS_DIGITS[max2_index], max2 * 100, LETTERS_DIGITS[max3_index], max3 * 100))

            print('车牌编号是：[%s]' % license_num)
            return license_num


if __name__ == '__main__':
    if find_car_num_brod():
        cut_car_num_for_chart()
        first = province_test()
        second = letter_test()
        last = last_5_num_test()
        print(first, second, last)
