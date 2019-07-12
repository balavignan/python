from keras.models import Sequential
from keras import layers
from keras.preprocessing.text import Tokenizer
from keras.layers import Embedding
from keras.preprocessing.sequence import pad_sequences
import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.layers import Flatten



df = pd.read_csv('imdb_master.csv',encoding='latin-1')
print(df.head())
sentences = df['review'].values
y = df['label'].values


max_review_len = max([len(s.split())for s in sentences])
#tokenizing data
tokenizer = Tokenizer(num_words=max_review_len)
tokenizer.fit_on_texts(sentences)

print(max_review_len)
#getting the vocabulary of data
sentences1 = tokenizer.texts_to_matrix(sentences)

le = preprocessing.LabelEncoder()
y = le.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(sentences1, y, test_size=0.25, random_state=1000)

# Number of features
input_dim = max_review_len
vocab_size = len(tokenizer.word_index)+1

sentences = tokenizer.texts_to_sequences(sentences)
padded_docs = pad_sequences(sentences, maxlen=max_review_len)

# print(input_dim)
model = Sequential()
model.add(Embedding(vocab_size, 50, input_length=max_review_len))
model.add(Flatten())
model.add(layers.Dense(300,input_dim=input_dim, activation='relu'))

#model.add(layers.Dense(30, activation='sigmoid'))

model.add(layers.Dense(3, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['acc'])
history=model.fit(X_train,y_train, epochs=2, verbose=True, validation_data=(X_test,y_test), batch_size=256)

#graph
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
print(history.history.keys())
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('accuracy')
plt.legend(['test', 'validation'], loc='upper left')
plt.show()
