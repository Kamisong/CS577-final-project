#This is the version with content and source as input, we delete the authors part.
import pandas as pd
from sklearn.model_selection import train_test_split
from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense,LSTM,Dropout, SimpleRNN
from keras.layers import Input
from keras.optimizers import Adam
import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
MAX_SEQ_LEN = 25000
input1 = Input(shape=(MAX_SEQ_LEN,))
input2 = Input(shape=(MAX_SEQ_LEN,))
# 读取CSV文件
data_train = pd.read_csv('Train.csv')
data_valid = pd.read_csv('Test.csv')
#data.SentimentText=data.SentimentText.astype(str)
# 提取输入特征和标签
X_train = data_train[['source', 'authors', 'content']][:8000]
X_val = data_valid[['source', 'authors', 'content']][:1000]
y_train = data_train['bias'][:8000]
y_val = data_valid['bias'][:1000]

# 将数据划分为训练集和验证集
#X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
# 对作者和新闻来源进行编码
tokenizer = Tokenizer()
X_train['authors'] = X_train['authors'].astype('str')
X_train['source'] = X_train['source'].astype('str')
X_train['content'] = X_train['content'].astype('str')
tokenizer.fit_on_texts( X_train['source'] + ' ' + X_train['content'])

X_train_authors = tokenizer.texts_to_sequences( X_train['source']+' '+X_train['content'])
X_val['authors'] = X_val['authors'].astype('str')
X_val['source'] = X_val['source'].astype('str')
X_val['content'] = X_val['content'].astype('str')
tokenizer.fit_on_texts(X_val['source'] + ' ' + X_val['content'])
X_val_authors = tokenizer.texts_to_sequences(X_val['source'] + ' ' + X_val['content'])

# 对新闻内容进行编码
#X_train['content'] = X_train['content'].astype('str')
#tokenizer.fit_on_texts(X_train['content'])
#X_train_content = tokenizer.texts_to_sequences(X_train['content'])
#X_val_content = tokenizer.texts_to_sequences(X_val['content'])

# 对输入特征进行填充，使其长度一致
max_len = 26000
X_train_authors = pad_sequences(X_train_authors, maxlen=max_len)
X_val_authors = pad_sequences(X_val_authors, maxlen=max_len)
#X_train_content = pad_sequences(X_train_content, maxlen=max_len)
#X_val_content = pad_sequences(X_val_content, maxlen=max_len)

# 构建卷积神经网络模型
model = Sequential()
model.add(Embedding(input_dim=100000, output_dim=32, input_length=max_len))
model.add(SimpleRNN(units=32))
model.add(Dense(units=3, activation='softmax'))

# 编译模型
with tf.device('/gpu:0'):
    model.compile(optimizer=Adam(lr=0.1), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 训练模型
    model.fit(X_train_authors, y_train, epochs=1, batch_size=128, validation_data=(X_val_authors, y_val))


# 使用测试集评估模型
X_test = pd.read_csv('Test.csv')
X_test['authors'] = X_test['authors'].astype('str')
X_test['source'] = X_test['source'].astype('str')
X_test['content'] = X_test['content'].astype('str')
X_test_authors = tokenizer.texts_to_sequences(X_test['source'] + ' ' + X_test['content'])
X_test_authors = pad_sequences(X_test_authors, maxlen=max_len)
y_test = X_test['bias']
_, accuracy = model.evaluate(X_test_authors, y_test)
print('Test accuracy:', accuracy)
