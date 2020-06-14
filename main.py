from nltk.stem.lancaster import LancasterStemmer
import json
import random
import tensorflow as tf
import tflearn
import numpy as np
import nltk
import pickle

nltk.download('punkt')
stemmer = LancasterStemmer()

# json para python doc
with open("intents.json") as file:
    data = json.load(file)

print(data["intents"])

# tentar abrir as variaveis salvas:
# try:
# with open("data.pickle", "rb") as f:
#words, labels, training, output = pickle.load(f)
# except:
words = []
labels = []
docs_x = []  # listas de todos os patterns
docs_y = []  # a tag de qual cada pattern faz parte

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        # adicionando os tokens das palavras
        wrds = nltk.word_tokenize(pattern)
        words.extend(wrds)
        docs_x.append(wrds)
        docs_y.append(intent["tag"])

    if intent["tag"] not in labels:
        labels.append(intent["tag"])

# aplicando a stemizacao
words = [stemmer.stem(w.lower()) for w in words]

words = sorted(list(set(words)))
labels = sorted(labels)

training = []
output = []  # lista do bow a ser preenchida
# iniciado com 0 em todas as posicoes da lista
out_empty = [0 for _ in range(len(labels))]

for x, doc in enumerate(docs_x):
    bag = []
    wrds = [stemmer.stem(w) for w in doc]

    for w in words:
        if w in wrds:
            bag.append(1)
        else:
            bag.append(0)

    output_row = out_empty[:]
    output_row[labels.index(docs_y[x])] = 1

    training.append(bag)
    output.append(output_row)

training = np.array(training)
output = np.array(output)
# salvando as variaveis para nao ter que executar tudo de novo
# with open("data.pickle", "wb") as f:
#pickle.dump((words, labels, training, output), f)

tf.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
# 2 layers com 8 neuronios
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
# probabilidade de cada neuronio
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
model.save("model.tflearn")


def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]
    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return np.array(bag)


def chat():
    print("Comece a falar com o bot")
    while True:
        inp = input("Voce > ")
        if inp.lower == "sair":
            break

        results = model.predict([bag_of_words(inp, words)])
        results_index = np.argmax(results)
        tag = labels[results_index]

        for tg in data["intents"]:
            if tg['tag'] == tag:
                responses = tg['responses']

        print(random.choice(responses))


chat()
