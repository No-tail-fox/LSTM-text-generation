import tensorflow as tf
import numpy as np
import tool

_, vocab_to_int, int_to_vocab, token_dict = tool.load_preprocess()

def get_tensors(loaded_graph):  #获取训练后的变量

    inputs = loaded_graph.get_tensor_by_name("inputs:0")    #输入变量

    initial_state = loaded_graph.get_tensor_by_name("init_state:0")   #未经训练的隐藏层

    final_state = loaded_graph.get_tensor_by_name("final_state:0") #训练过后的隐藏层

    probs = loaded_graph.get_tensor_by_name("probs:0") #这个是已经进行过softmax处理的输出层

    return inputs, initial_state, final_state, probs

def pick_word(probabilities, int_to_vocab):

    chances = []
    for idx, prob in enumerate(probabilities): #这里由于probabilities本身就是一个长度为3436的一维数组，每一个维度正好与int_to_vocab一一对应
        if prob >= 0.02:
            chances.append(int_to_vocab[idx])
    rand = np.random.randint(0, len(chances))
    return str(chances[rand])

gen_length = 1000  #生成的文本长度

# 文章开头的字，指定一个即可，这个字必须是在训练词汇列表中的
prime_word = '叶'

seq_length=30 #一行字符串的长度

loaded_graph = tf.Graph()
with tf.Session(graph=loaded_graph) as sess:
    # 加载保存过的session
    loader = tf.train.import_meta_graph('./save'+'.meta') #这里是之前保存的模型名+ .meta
    loader.restore(sess, './save')  #加载模型

    # 通过名称获取缓存的tensor
    input_text, initial_state, final_state, probs = get_tensors(loaded_graph) #获取之前的名字

    # 准备开始生成文本
    gen_sentences = ['天','地','不','仁']
    prev_state = sess.run(initial_state, {input_text: np.array([[1]])}) #

    # 开始生成文本
    for n in range(gen_length): #开始生成文本了
        dyn_input = [[vocab_to_int[word] for word in gen_sentences[-seq_length:]]]  #[-seq_length：]会永远获得全部值,又因为input_text为两维，所以这个在外层还要加一个括号
        dyn_seq_length = len(dyn_input[0])   #dyn_input 为(1,30) 二维数组
        probabilities, prev_state = sess.run(
            [probs, final_state],
            {input_text: dyn_input, initial_state: prev_state})
        pred_word = pick_word(probabilities[0][dyn_seq_length - 1], int_to_vocab) #这里probabilities正好是三维矩阵，(1,30,3436)
        gen_sentences.append(pred_word)

    # 将标点符号还原
    novel = ''.join(gen_sentences)
    for key, token in token_dict.items():
        ending = ' ' if key in ['\n', '（', '“'] else ''
        novel = novel.replace(token, key)
    novel = novel.replace('\n ', '\n')
    novel = novel.replace('（ ', '（')

    print(novel)