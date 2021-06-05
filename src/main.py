import numpy as np
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.models import Model

import data


BITS = 3

inputs = Input(shape=(2, BITS))
x = Flatten()(inputs)
x = Dense(2 ** (2 * BITS + 2), activation='sigmoid')(x)
bit_layers = [Dense(1, activation='sigmoid', name='b_{}'.format(str(i)))(x) for i in range(BITS)]

model = Model(inputs=inputs, outputs=bit_layers)
model.summary()

model.compile(optimizer='rmsprop',
              loss=['binary_crossentropy']*BITS,
              metrics=[['accuracy']]*BITS)

data, labels = data.load_data()

ex = (lambda f: (f(data[0][0]), f(data[0][1]), f(labels[0][0])))(lambda l: ''.join([str(int(b)) for b in l]))
print('{0} ({3}) + {1} ({4}) = {2} ({5})'.format(*ex, *[int(n, 2) for n in ex]))

labels = np.moveaxis(labels, -1, 0)
labels = [labels[i] for i in range(BITS)]

model.fit(
    data,
    labels,
    epochs=5, verbose=1,
    validation_split=0.2,
    shuffle=True)

print(model.predict(np.array([[[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]]])))
