import tool

def create_lookup_tables(input_data):  #创建数字到字符和字符到数字的映射
    vocab=set(input_data) #生成无序不重复列表
    vocab_to_int={word:index for index,word in enumerate(vocab)}#enumerate 返回索引和对应的字符
    int_to_vocab={index:word for index,word in enumerate(vocab)}
    return vocab_to_int,int_to_vocab

def token_lookup(): #建立标点符号的映射表
    symbols = set(['。', '，', '“', "”", '；', '！', '？', '（', '）', '——', '\n'])
    tokens = ["P", "C", "Q", "T", "S", "E", "M", "I", "O", "D", "R"]
    return dict(zip(symbols,tokens))

file="data/遮天.txt"
text=tool.load_text(file)

num_words_for_training = 500000 #设置用多少数据进行训练
text=text[:num_words_for_training]

line_of_text=text.split("\n") #切割文本，有17637行文本

line_of_text=[lines for lines in line_of_text if len(lines)>0] #去除了空白行，只有8819行了
line_of_text=line_of_text[1:]
line_of_text=[lines.strip() for lines in line_of_text]  #去除了首尾的空白
line_of_text=[lines for lines in line_of_text if lines.find('※')==-1] #数据处理结束后，一共有8813行

vocab_to_int,int_to_vocab=create_lookup_tables(''.join(line_of_text))
token_tables=token_lookup()

tool.preprocess_and_save_data(''.join(line_of_text), token_tables, create_lookup_tables)


