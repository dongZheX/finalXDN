import numpy as np
import tensorflow as tf
import pandas as pd
#数据处理函数
# 处理行X14
def func1(x):
    if x == "100(AO)":
        return 1
    elif x == "100(CBO)" :
        return 2
    elif x== "80(CBO):20(AO)":
        return 3
    elif x== "40(CBO):60(AO)":
        return 4
    elif x == "50(CBO):50(AO)":
        return 5
    elif x== "60(CBO):40(AO)":
        return 6
    elif x== "70(CBO):30(AO)":
        return 7
    elif x == "80(CBO):20(AO)":
        return 8
# 处理行20

def func2(x):
    if x == "16*0.62":
        return  1
    else:
        return 2
# 处理行24

def func3(x):
    if x == "24*0.086":
        return 1
    elif x == "36*0.080":
        return 2
    elif x== "36*0.086":
        return 3
    elif x=="BM32":
        return 4
def func4(x):
    x = x.split("_")
    return int(x[1])
# 导入数据
data = pd.read_excel("data1.xlsx")
data['X36_VALUE'] = data['X36_VALUE'].apply(lambda x: 1 if x == 'Open' else 0)
data['X35_VALUE'] = data['X35_VALUE'].apply(lambda x: 1 if x == 'Open' else 0)
data['X32_VALUE'] = data['X32_VALUE'].apply(lambda x: 1 if x == 'Delavan 25GPM' else 0)
data['X14_VALUE'] = data['X14_VALUE'].apply(func1)
data['X20_VALUE'] = data['X20_VALUE'].apply(func2)
data['X24_VALUE'] = data['X24_VALUE'].apply(func3)
data['PRODUCT'] = data['PRODUCT'].apply(func4)
data = data.fillna(0)
data = data[data['PRODUCT']==1]
print(data)
data_x = data[['Y0_VALUE','Y1_VALUE','Y2_VALUE','Y3_VALUE','Y4_VALUE','Y5_VALUE','Y6_VALUE','Y7_VALUE','Y8_VALUE','Y9_VALUE','Y10_VALUE']]
arr = []
for i in range(1, 38):
    arr.append('X%d_VALUE' % i)
data_y = data[arr]
data_x = np.array(data_x)
data_y = np.array(data_y)
np.set_printoptions(threshold=np.inf)
data_x = data_x.astype(np.float32)
data_y = data_y.astype(np.float32)
# 归一化
datax_max = np.max(data_x, axis=0)
datax_min = np.min(data_x, axis=0)
datay_max = np.max(data_y, axis=0)
datay_min = np.min(data_y, axis=0)
data_x = (data_x-datax_min)/(datax_max-datax_min+0.0001)
data_y = (data_y-datay_min)/(datay_max-datay_min+0.0001)
toSave = np.array([])
np.save("./datay_max",datay_max)
np.save("./datay_min",datay_min)

datay_min.tofile("datay_min")
# 构建网络
# 定义占位符
xt = tf.placeholder(tf.float32,[None,11],name='x')
yt = tf.placeholder(tf.float32,[None,37])
# 定义添加层
in_size = 11
out_size = 37
def add_layer(inputs, in_size, out_size, use_relu):
    w = tf.Variable(tf.truncated_normal([in_size, out_size]))
    b = tf.Variable(tf.zeros(out_size))
    out = tf.add(tf.matmul(inputs, w),b)
    if use_relu:
        out = tf.nn.sigmoid(out)
    return out
h1 = add_layer(xt, in_size, 200, True)
h2 = add_layer(h1, 200, 100, True)
h3 = add_layer(h2, 100, 50, True)
#h4 = add_layer(h3, 200, 100, True)
pred1 = add_layer(h3, 50, out_size, False)
pred = tf.nn.sigmoid(pred1)
#pred = tf.nn.sigmoid(pred2)
tf.add_to_collection('pred',pred)
# 损失函数 均方误差绝对值
loss = tf.reduce_mean(tf.reduce_sum(tf.square(data_y - pred), reduction_indices=[1]))
# 梯度下降 优化函数 学习率
train_step = tf.train.AdamOptimizer(0.001).minimize(loss=loss)

# 进行训练
with tf.Session() as sess:
    initializer = tf.global_variables_initializer()
    MYLOSS = np.zeros(5000)
    sess.run(initializer)
    for i in range(5000):
        sess.run(train_step, feed_dict={xt: data_x, yt: data_y})
        MYLOSS[i] = sess.run(loss, feed_dict={xt: data_x, yt: data_y })
        print(MYLOSS[i])
    saver = tf.train.Saver()
    save_path = saver.save(sess,"./factory.ckpt")
