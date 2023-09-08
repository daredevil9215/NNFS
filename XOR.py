from objekt import *

X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

y = np.array([[0], [1], [1], [0]])

model = Model()

model.add(Layer_Dense(2, 16))
model.add(Activation_Sigmoid())
model.add(Layer_Dense(16, 1))
model.add(Activation_Sigmoid())

model.set(loss=Loss_MeanSquaredError(), optimizer=Optimizer_Adam(learning_rate=0.05,decay=5e-4), accuracy=Accuracy_Regression())

model.finalize()

model.train(X, y, epochs=500, print_every=100)

output = model.predict(X)
prediction = model.output_layer_activation.predictions(y)
print(output)
print(prediction)

