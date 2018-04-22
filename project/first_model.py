import tensorflow as tf
import numpy as np
import tool
from tensorflow.contrib import seq2seq

# 训练循环次数
num_epochs = 20 #训练的次数

batch_size=256 #单个batch中字符串的行数

rnn_size=513 #单个lstm层中神经元的个数

embed_dim = 512 #嵌入层的神经元个数

seq_length=30 #单个字符串的长度

# 每多少步打印一次训练信息
show_every_n_batches = 2

learning_rate=0.006 #学习率

save_dir = './save2' # 保存session状态的位置

int_text, vocab_to_int, int_to_vocab, token_dict = tool.load_preprocess() #int_text是映射成数字的小说

def build_inputs(): #输入层
    inputs=tf.placeholder(tf.int32,shape=(None,None),name='inputs') #输入的值，设置成占位符
    targets=tf.placeholder(tf.int32,shape=(None,None),name='targets') #目标值，设置成占位符

    keep_drop=tf.placeholder(tf.float32,name='keep_drop') #设置的dropout的概率值,正则化使用
    return inputs,targets,keep_drop

def build_lstm(batch_size,lstm_size): #lstm隐藏层

    num_lays=2 #隐藏层的层数

    keep_drop=0.8

    lstm=tf.contrib.rnn.BasicLSTMCell(lstm_size) #创建基本lstm单元

    drop=tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_drop) #创建dropout层

    lstm= tf.contrib.rnn.MultiRNNCell([drop for _ in range(num_lays)]) #创建多层lstm层

    init_state = lstm.zero_state(batch_size, tf.float32) #定义隐藏层的初始化状态

    # 使用tf.identify给init_state取个名字，后面生成文字的时候，要使用这个名字来找到缓存的state
    init_state = tf.identity(init_state, name='init_state')  #只有这样，run的时候才能找到

    return lstm,init_state

def build_embed(input_data,vocab_size,embed_dim): #嵌入层，提升效率，重点

    # 先根据文字数量和embedding layer的size创建tensorflow variable
    embedding = tf.Variable(tf.random_uniform((vocab_size, embed_dim)), dtype=tf.float32)

    # 让tensorflow帮我们创建lookup table
    return tf.nn.embedding_lookup(embedding, input_data)

def build_rnn(cell, inputs): #cell是RNN的节点，inputs是字符串的长度
    outputs, final_state = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32) #一种方便的处理，可以避免填充
    # 同样给final_state一个名字，后面要重新获取缓存
    final_state = tf.identity(final_state, name="final_state")
    return outputs, final_state

def build_outputs(cell, rnn_size, input_data, vocab_size, embed_dim): #创建输出层

    # 创建embedding layer
    embed = build_embed(input_data, vocab_size, rnn_size)

    # 计算outputs 和 final_state
    outputs, final_state = build_rnn(cell, embed)

    # remember to initialize weights and biases, or the loss will stuck at a very high point
    logits = tf.contrib.layers.fully_connected(outputs, vocab_size, activation_fn=None,
                                               weights_initializer = tf.truncated_normal_initializer(stddev=0.1),
                                               biases_initializer=tf.zeros_initializer())
    return logits, final_state

def get_batches(int_text, batch_size, seq_length): #获取单个的batch，int_text为总字符列表，batch_size为 一个batch有多少行字符串，seq_length表示一行有多少个字符

    # 计算有多少个batch可以创建
    n_batches = (len(int_text) // (batch_size * seq_length))  #计算一共有多少个batch #这里没问题，MD //代表的是去整除

    # 计算每一步的原始数据，和位移一位之后的数据
    batch_origin = np.array(int_text[: n_batches * batch_size * seq_length]) #构成矩阵
    batch_shifted = np.array(int_text[1: n_batches * batch_size * seq_length + 1]) #构成矩阵

    # 将位移之后的数据的最后一位，设置成原始数据的第一位，相当于在做循环
    batch_shifted[-1] = batch_origin[0]  #！！！！需要思考

    batch_origin_reshape = np.split(batch_origin.reshape(batch_size, -1), n_batches, 1)  #reshape 设置成 batch 行的矩阵，split将这个矩阵重新划分 成n_batches个，同时是按照列的!!!!重点
    batch_shifted_reshape = np.split(batch_shifted.reshape(batch_size, -1), n_batches, 1)

    batches = np.array(list(zip(batch_origin_reshape, batch_shifted_reshape)))  #list 生成数组，也算是矩阵

    return batches

train_graph = tf.Graph()    #创建一个图
with train_graph.as_default():  #把创建的图设为默认

    vocab_size = len(int_to_vocab) #获得文字总量

    # 获取模型的输入，目标以及学习率节点，这些都是tf的placeholder
    input_text, targets, lr = build_inputs()

    # 输入数据的shape
    input_data_shape = tf.shape(input_text) #获得这个位置变量的shape

    cell, initial_state = build_lstm(input_data_shape[0], rnn_size) #这个input_data_shape[0] 指的是batch_size 也就是一个batch里有几行字符串

    logits, final_state=build_outputs(cell,rnn_size,input_text,vocab_size,embed_dim)  #这里的rnnsize为 神经元个数

    out_vlaue=tf.nn.softmax(logits, name='probs')

    cost = seq2seq.sequence_loss(
        logits,
        targets,
        tf.ones([input_data_shape[0], input_data_shape[1]]))  #损失函数

    optimizer = tf.train.AdamOptimizer(lr)  #梯度下降
    gradients = optimizer.compute_gradients(cost)   # 裁剪一下Gradient输出，最后的gradient都在[-1, 1]的范围内
    capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if grad is not None]
    train_op = optimizer.apply_gradients(capped_gradients)


#训练模型
batches = get_batches(int_text, batch_size, seq_length)  #获取所有的mini数据集

# 打开session开始训练，将上面创建的graph对象传递给session
with tf.Session(graph=train_graph) as sess:
    sess.run(tf.global_variables_initializer())  #使所有变量定义

    for epoch_i in range(num_epochs):
        state = sess.run(initial_state, {input_text: batches[0][0]}) #第一组数据集的训练数据
        for batch_i, (x, y) in enumerate(batches): #这时候，x为训练值，y为真实值，initial_state为神经元的初始状态
            feed = {
                input_text: x,
                targets: y,
                initial_state: state,
                lr: learning_rate}
            train_loss, state, _ = sess.run([cost, final_state, train_op], feed)
            logits_value=sess.run([logits],feed)

            print('soft:{}'.format(logits_value))
            # 打印训练信息
            if (epoch_i * len(batches) + batch_i) % show_every_n_batches == 0:
                print('Epoch {:>3} Batch {:>4}/{}   train_loss = {:.3f}'.format(
                    epoch_i,
                    batch_i,
                    len(batches),
                    train_loss))

                # 保存模型
    saver = tf.train.Saver()
    saver.save(sess, save_dir)
    print('Model Trained and Saved')