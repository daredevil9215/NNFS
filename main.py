from objekt import *
import nnfs
nnfs.init()
FOLDER = '/home/play/Desktop/TVZ/Python_NNFS/main/fashion_mnist_images'

X, y, X_test, y_test = create_mnist_dataset(FOLDER)

keys = np.array(range(X.shape[0]))
np.random.shuffle(keys)
X = X[keys]
y = y[keys]

X = (X.reshape(X.shape[0], -1).astype(np.float32) - 127.5) / 127.5
X_test = (X_test.reshape(X_test.shape[0], -1).astype(np.float32) - 127.5) / 127.5


model = Model()

model.add(Layer_Dense(X.shape[1], 64))
model.add(Activation_ReLU())
model.add(Layer_Dense(64, 64))
model.add(Activation_ReLU())
model.add(Layer_Dense(64, 10))
model.add(Activation_Softmax())

model.set(loss=Loss_CCE(), optimizer=Optimizer_Adam(learning_rate=0.001, decay=1e-4), accuracy=Accuracy_Categorical())

model.finalize()

model.train(X, y, validation_data=(X_test, y_test), batch_size=128, epochs=50, print_every=100)

model.save('/home/play/Desktop/TVZ/Python_NNFS/main/fashion_mnist.model')

model.evaluate(X_test, y_test)



