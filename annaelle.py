import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.utils import losses_utils
import matplotlib.pyplot as plt

acids_indices = {
    'A': 0,
    'C': 1,
    'D': 2,
    'E': 3,
    'F': 4,
    'G': 5,
    'H': 6,
    'I': 7,
    'K': 8,
    'L': 9,
    'M': 10,
    'N': 11,
    'P': 12,
    'Q': 13,
    'R': 14,
    'S': 15,
    'T': 16,
    'V': 17,
    'W': 18,
    'Y': 19
}


class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.d1 = Dense(1000, activation='relu')
        self.d2 = Dense(200, activation='relu')
        self.d3 = Dense(800, activation='relu')
        self.d4 = Dense(200, activation='relu')
        self.d5 = Dense(800, activation='relu')
        self.d6 = Dense(1)

    def call(self, x):
        x = self.d1(x)
        x= self.d2(x)
        x = self.d3(x)
        x = self.d4(x)
        x = self.d5(x)


        return self.d6(x)

    def loss(self, y, y_pred):


@tf.function
def train_step(data, labels):
    with tf.GradientTape() as tape:
        # training=True is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = model(data, training=True)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)


@tf.function
def test_step(images, labels):
    # training=False is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    predictions = model(images, training=False)
    t_loss = loss_object(labels, predictions)

    test_loss(t_loss)
    test_accuracy(labels, predictions)


def preprocess_data(filename, label):
    with open(filename) as f:
        lines = f.read().splitlines()

    mat = np.zeros((len(lines), 181))

    for i, line in enumerate(lines):
        for j, a in enumerate(line):
            mat[i, 20 * j + acids_indices[a.upper()]] = 1.

    mat[:, -1] = label

    return mat


def load_data():
    neg_mat = preprocess_data("neg_A0201.txt", 0.)
    pos_mat = preprocess_data("pos_A0201.txt", 1.)

    data_mat = np.concatenate([neg_mat, pos_mat], axis=0)
    np.random.shuffle(data_mat)

    return data_mat[:, :-1], data_mat[:, -1]

# def plot(arr_loss_test)

def plot_graphs():
    plt.figure()
    plt.title('loss')
    plt.plot(np.arange(1, EPOCHS + 1), arr_loss_test, label='test')
    plt.plot(np.arange(1, EPOCHS + 1), arr_loss_train, label='train')
    plt.xlabel("EPOCH NUM")
    plt.ylabel('loss')
    plt.legend()
    plt.figure()
    plt.title('accuracy')
    plt.plot(np.arange(1, EPOCHS + 1), arr_accuracy_test, label='test')
    plt.plot(np.arange(1, EPOCHS + 1), arr_accuracy_train, label='train')
    plt.xlabel("EPOCH NUM")
    plt.ylabel('accuracy')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    data, labels = load_data()
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.1)
    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(10000).batch(50)

    test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(50)
    model = MyModel()
    loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam()
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.BinaryAccuracy(name='train_accuracy')

    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.BinaryAccuracy(name='test_accuracy')

    EPOCHS = 10
    arr_accuracy_test = [0] * EPOCHS
    arr_loss_test = [0] * EPOCHS
    arr_accuracy_train = [0] * EPOCHS
    arr_loss_train = [0] * EPOCHS

    for epoch in range(EPOCHS):
        # Reset the metrics at the start of the next epoch
        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()

        for images, labels in train_ds:
            train_step(images, labels)

        for test_images, test_labels in test_ds:
            test_step(test_images, test_labels)

        print(
            f'Epoch {epoch + 1}, '
            f'Loss: {train_loss.result()}, '
            f'Accuracy: {train_accuracy.result() * 100}, '
            f'Test Loss: {test_loss.result()}, '
            f'Test Accuracy: {test_accuracy.result() * 100}'
        )
        arr_accuracy_test[epoch] = test_accuracy.result() * 100
        arr_loss_test[epoch] = test_loss.result()
        arr_accuracy_train[epoch] = train_accuracy.result() * 100
        arr_loss_train[epoch] = train_loss.result()

    plot_graphs()
