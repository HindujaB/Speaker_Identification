# Dependencies
from keras.layers import Dense
from keras.models import Sequential


# Neural network
def model():
    model = Sequential()
    model.add(Dense(16, input_dim=39, activation='relu'))
    model.add(Dense(12, activation='relu'))
    model.add(Dense(4, activation='softmax'))

    print('model summary : ', model.summary())
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def train_network(self, training_set, class_set):
    history = model.fit(training_set, class_set, epochs=100, batch_size=64)

# TODO: change NN structure with number of classes and implement plots and save models
