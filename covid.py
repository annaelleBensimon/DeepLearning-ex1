import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score

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

EPOCHS = 10

#
# class AnglesLoss(Loss):
#   def call(self, y_true, y_pred):
#     return tf.square(tf.sin(0.5 * (y_true- y_pred)))

class Model1(Model):
    def __init__(self):
        super(Model1, self).__init__()
        self.d1 = Dense(512, activation='relu')
        self.d2 = Dense(512, activation='relu')
        self.d3 = Dense(256, activation='relu')
        self.d4 = Dense(1)

    def call(self, x):
        x = self.d1(x)
        x = self.d2(x)
        x = self.d3(x)
        return self.d4(x)


class Model2(Model):
    def __init__(self):
        super(Model2, self).__init__()
        self.d1 = Dense(512, activation='relu')
        self.d2 = Dense(512, activation='relu')
        self.d3 = Dense(256, activation='relu')
        self.d4 = Dense(256, activation='relu')
        self.d5 = Dense(256, activation='relu')
        self.d6 = Dense(1)

    def call(self, x):
        x = self.d1(x)
        x = self.d2(x)
        x = self.d3(x)
        x = self.d4(x)
        x = self.d5(x)
        return self.d6(x)


class Model3(Model):
    def __init__(self):
        super(Model3, self).__init__()
        self.d1 = Dense(512, activation='sigmoid')
        self.d2 = Dense(512, activation='sigmoid')
        self.d3 = Dense(256, activation='sigmoid')
        self.d4 = Dense(1)

    def call(self, x):
        x = self.d1(x)
        x = self.d2(x)
        x = self.d3(x)
        return self.d4(x)


class Model4(Model):
    def __init__(self):
        super(Model4, self).__init__()
        self.d1 = Dense(512, activation='sigmoid')
        self.d2 = Dense(512, activation='sigmoid')
        self.d3 = Dense(256, activation='sigmoid')
        self.d4 = Dense(256, activation='sigmoid')
        self.d5 = Dense(256, activation='sigmoid')
        self.d6 = Dense(1)

    def call(self, x):
        x = self.d1(x)
        x = self.d2(x)
        x = self.d3(x)
        x = self.d4(x)
        x = self.d5(x)
        return self.d6(x)


class Model5(Model):
    def __init__(self):
        super(Model5, self).__init__()
        self.d1 = Dense(20, activation='relu')
        self.d2 = Dense(10, activation='relu')
        self.d3 = Dense(1)

    def call(self, x):
        x = self.d1(x)
        x = self.d2(x)
        return self.d3(x)


class Model6(Model):
    def __init__(self):
        super(Model6, self).__init__()
        self.d1 = Dense(20, activation='sigmoid')
        self.d2 = Dense(10, activation='sigmoid')
        self.d3 = Dense(1)

    def call(self, x):
        x = self.d1(x)
        x = self.d2(x)
        return self.d3(x)


def create_models():
    return [Model1(), Model2(), Model3(), Model4(), Model5(), Model6()]


def preprocess_data(filename, label):
    with open(filename) as f:
        lines = f.read().splitlines()

    mat = np.zeros((len(lines), 180))

    for i, line in enumerate(lines):
        for j, a in enumerate(line):
            mat[i, 20 * j + acids_indices[a.upper()]] = 1.

    return mat, label * np.ones(mat.shape[0])


def load_data():
    neg_mat, neg_labels = preprocess_data("neg_A0201.txt", 0.)
    pos_mat, pos_labels = preprocess_data("pos_A0201.txt", 1.)

    X_neg_train, X_neg_test, y_neg_train, y_neg_test = train_test_split(neg_mat, neg_labels, test_size=0.1)
    X_pos_train, X_pos_test, y_pos_train, y_pos_test = train_test_split(pos_mat, pos_labels, test_size=0.1)

    X_pos_train = np.tile(X_pos_train, (len(X_neg_train) // len(X_pos_train), 1))
    y_pos_train = np.tile(y_pos_train, (len(y_neg_train) // len(y_pos_train)))

    X_train = np.concatenate([X_neg_train, X_pos_train])
    y_train = np.concatenate([y_neg_train, y_pos_train])
    X_test = np.concatenate([X_neg_test, X_pos_test])
    y_test = np.concatenate([y_neg_test, y_pos_test])

    return X_train, y_train, X_test, y_test


def plot_graphs(model_index, arr_loss_train, arr_accuracy_train, arr_loss_test, arr_accuracy_test):
    plt.figure()
    plt.title('Model ' + str(model_index+1) + ' Train and Test Loss')
    plt.plot(np.arange(1, EPOCHS + 1), arr_loss_train, label='train')
    plt.plot(np.arange(1, EPOCHS + 1), arr_loss_test, label='test')
    plt.xlabel("epoch number")
    plt.ylabel('loss')
    plt.legend()
    plt.savefig('model' + str(model_index+1) + '_loss.png')

    plt.figure()
    plt.title('Model ' + str(model_index+1) + ' Train and Test Accuracy')
    plt.plot(np.arange(1, EPOCHS + 1), arr_accuracy_train, label='train')
    plt.plot(np.arange(1, EPOCHS + 1), arr_accuracy_test, label='test')
    plt.xlabel("epoch number")
    plt.ylabel('accuracy')
    plt.legend()
    plt.savefig('model' + str(model_index+1) + '_accuracy.png')

    # plt.show()


def load_data_covid(name_file):
    with open(name_file) as f:
        f.readline()

        str_rows = ''.join(f.read().splitlines())

    data = []
    for i in range(0, len(str_rows), 8):
        data.append(str_rows[i:i+9])
    data.pop()
    mat = np.zeros((len(data), 180))

    for i, line in enumerate(data):
        for j, a in enumerate(line):
            mat[i, 20 * j + acids_indices[a.upper()]] = 1.

    return mat, data


def train_model(model, model_index, num_epochs, X_train, y_train, X_test, y_test):
    loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam()

    @tf.function
    def train_step(data, labels):
        with tf.GradientTape() as tape:
            predictions = model(data, training=True)
            loss = loss_object(labels, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss(loss)
        train_accuracy(labels, predictions)

    @tf.function
    def test_step(data, labels):
        predictions = model(data, training=False)
        t_loss = loss_object(labels, predictions)

        test_loss(t_loss)
        test_accuracy(labels, predictions)

    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(10000).batch(32)
    test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)).shuffle(10000).batch(32)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.BinaryAccuracy(name='train_accuracy')
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.BinaryAccuracy(name='test_accuracy')

    arr_accuracy_test = [0] * num_epochs
    arr_loss_test = [0] * num_epochs
    arr_accuracy_train = [0] * num_epochs
    arr_loss_train = [0] * num_epochs

    best_model_weights = None
    best_accuracy = 0
    best_epoch = 0

    print("Training Model " + str(model_index+1) + ":")

    for epoch in range(num_epochs):
        # Reset the metrics at the start of the next epoch
        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()

        for batch, labels in train_ds:
            train_step(batch, labels)

        for test_batch, test_labels in test_ds:
            test_step(test_batch, test_labels)

        print(
            f'Epoch {epoch + 1}, '
            f'Loss: {train_loss.result()}, '
            f'Accuracy: {train_accuracy.result() * 100}, '
            f'Test Loss: {test_loss.result()}, '
            f'Test Accuracy: {test_accuracy.result() * 100}'
            # f'Test Recall: {recall_score(y_test,model.predict(X_test)) * 100}'
            # f'Test Precision: {precision_score(y_test,np.array(model.predict(X_test))) * 100}'
        )
        arr_loss_train[epoch] = train_loss.result()
        arr_accuracy_train[epoch] = train_accuracy.result() * 100
        arr_loss_test[epoch] = test_loss.result()
        arr_accuracy_test[epoch] = test_accuracy.result() * 100

        if test_accuracy.result() > best_accuracy:
            best_accuracy = test_accuracy.result()
            best_model_weights = model.get_weights()
            best_epoch = epoch

    print("Best model from epoch", best_epoch + 1, "with test accuracy:", best_accuracy * 100)
    model.set_weights(best_model_weights)
    plot_graphs(model_index, arr_loss_train, arr_accuracy_train, arr_loss_test, arr_accuracy_test)

    return best_accuracy


if __name__ == '__main__':
    all_data = load_data()
    models = create_models()

    best_model_index = -1
    best_model_accuracy = 0

    for i, model in enumerate(models):
        model_accuracy = train_model(model, i, EPOCHS, *all_data)
        if model_accuracy > best_model_accuracy:
            best_model_index = i
            best_model_accuracy = model_accuracy

        print("********************************************************************************")
        print()

    best_model = models[best_model_index]
    print("Best Model is: Model", best_model_index+1, "with accuracy:", best_model_accuracy)

    mat_covid, data_covid = load_data_covid('P0DTC2.fasta')
    covid_labels = np.array(best_model.predict(mat_covid))
    top_5_positive = covid_labels.flatten().argsort()[-5:][::-1]

    print("Top 5 positive labeled examples:")
    for i in top_5_positive:
        print(data_covid[i] + " with score " + str(covid_labels[i]))
