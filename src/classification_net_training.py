import numpy as np
from keras.models import Model
from keras.layers import Dense,BatchNormalization,Conv1D, LSTM, ReLU
from keras.layers import Input,GlobalMaxPooling1D, GlobalAveragePooling1D, concatenate, AveragePooling1D, Flatten
from keras.optimizers import Adam, SGD
from utils import generate
import tensorflow as tf
from keras.callbacks import EarlyStopping,ReduceLROnPlateau,ModelCheckpoint,CSVLogger
import datetime
from keras.callbacks import History
import matplotlib.pyplot as plt
import pickle
from residual_block import residual_block
from blurpool import AverageBlurPooling1D

history = History()

batchsize = 128
T = np.arange(19,21,0.1) # this provides another layer of stochasticity to make the network more robust
steps = 30
 # number of steps to generate
initializer = 'he_normal'
f = 32 #number of filters
sigma = 0 #noise variance
EPOCHS = 25

inputs = Input((steps-1,1))

x1 = Conv1D(64,1, kernel_initializer=initializer, bias=False)(inputs)
x1 = BatchNormalization()(x1)
x2 = residual_block(x1, 64)
x3 = Conv1D(128,1, kernel_initializer=initializer)(x2)
x3 = residual_block(x3, 128)
x4 = Conv1D(256,1, kernel_initializer=initializer)(x3)
x4 = residual_block(x4, 256, d=2)
x5 = Conv1D(512,1, kernel_initializer=initializer)(x4)
x5 = residual_block(x5, 512, d=2)

# y1 = GlobalAveragePooling1D()(x1)
# y2 = GlobalAveragePooling1D()(x2)
# y3 = GlobalAveragePooling1D()(x3)
# y4 = GlobalAveragePooling1D()(x4)
y5 = GlobalAveragePooling1D()(x5)

# y2 = Flatten()(x2)
# y3 = Flatten()(x3)
# y4 = Flatten()(x4)
# y5 = Flatten()(x5)
# con = concatenate([y2, y3, y4, y5])

dense2 = Dense(1024,activation='relu')(y5)
# dense = Dense(512,activation='relu')(dense)
# # dense = Dense(128,activation='relu')(dense)
# dense = Dense(64,activation='relu')(dense)
# dense = Dense(32,activation='relu')(dense)
dense3 = Dense(3,activation='softmax')(dense2)
model = Model(inputs=inputs, outputs=dense3) 

# optimizer = SGD(lr=0.01, momentum=0.01, nesterov=False)
optimizer = Adam(lr=1e-5)
model.compile(optimizer=optimizer,loss='categorical_crossentropy',metrics=['acc'])
model.summary()


callbacks = [
         ReduceLROnPlateau(monitor='val_loss',
                           factor=0.1,
                           patience=3,
                           verbose=1,
                           min_lr=1e-9),
         ModelCheckpoint(filepath=f'./Models/{steps}_model.h5',
                         monitor='val_acc',
                         save_best_only=False,
                         mode='max',
                         save_weights_only=False), history]


gen = generate(batchsize=batchsize,steps=steps,T=T,sigma=sigma)
model.fit_generator(generator=gen,
        steps_per_epoch=50,
        epochs=EPOCHS,
        verbose=1,
        callbacks=callbacks,
        validation_data=generate(batchsize=batchsize,steps=steps,T=T,sigma=sigma),
        validation_steps=25)


train_loss = history.history['loss']
val_loss   = history.history['val_loss']
train_acc  = history.history['acc']
val_acc    = history.history['val_acc']
xc         = range(EPOCHS)


##https://stackoverflow.com/questions/11026959/writing-a-dict-to-txt-file-and-reading-it-back
with open(f'maxpoolingthreedense/{steps}_history.txt', 'wb') as handle:
    pickle.dump(history.history, handle)

plt.figure()
plt.plot(xc, train_loss, label="train loss")
plt.plot(xc, val_loss, label="validation loss")
plt.legend()
plt.savefig(f"maxpoolingthreedense/{steps}_loss.png")