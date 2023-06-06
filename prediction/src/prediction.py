import numpy as np
import time
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.keras import activations
#  import zinc_grammar
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import h5py
import os
from utils import info_all_data
from hyper_parse import parser

args = parser.parse_args()


os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.GPU}"

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

train = args.train
trainset  = args.trainset
learning_rate = args.LR
save_loss =True

#  data_name = f'S0'
with h5py.File(f'{trainset}','r') as h5f:
    data = h5f['train_data'][:]
    ans  = h5f['train_ans'][:]
    test      = h5f['test_data'][:]
    test_ans  = h5f['test_ans'][:]
    h5f.close()

print("dataset_loaded")

dataset = tf.data.Dataset.from_tensor_slices((data,ans))
dataset = dataset.shuffle(buffer_size=1024).batch(32)

test = tf.data.Dataset.from_tensor_slices((test,test_ans))
test = test.batch(1)

MAX_LEN, NCHARS = data.shape[1], data.shape[2]

model = keras.Sequential()

model.add(layers.LSTM(128,
    input_shape=(MAX_LEN,NCHARS),
    activation='tanh',
    return_sequences=True)
    )
model.add(layers.Dropout(0.2))

model.add(layers.LSTM(128,
    activation='tanh',
    return_sequences=False)
    )
model.add(layers.Dropout(0.2))

model.add(layers.Dense(64,
    activation = 'relu')
    )
model.add(layers.Dense(32,
    activation='relu')
    )

model.add(layers.Dense(16,
    activation='relu')
    )

model.add(layers.Dense(8,
    activation='relu')
    )

model.add(layers.Dense(1))

model.summary()
model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))
early_stop = EarlyStopping(monitor='val_loss', patience=15000)

model_path = 'model'
filename = os.path.join(model_path, f'{trainset}_check.h5')
checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=100, save_best_only=False, mode='auto')

if train:
    t_start = time.perf_counter()
    history = model.fit_generator(dataset,
                                        epochs=30000,
                                        validation_data=(test),
                                        use_multiprocessing=True,
                                        workers=4,
                                        callbacks=[checkpoint,early_stop])
    t_end = time.perf_counter()

else: pass

if save_loss and train:
    np_loss = np.array(history.history['loss'])
    np.savetxt(f'{data_name}_loss.txt',np_loss)
    np_val_loss = np.array(history.history['val_loss'])
    np.savetxt(f'{data_name}_val_loss.txt',np_val_loss)


model.load_weights(filename)

data = []
results = []
MSE = []
MAPE = []
for test_ans in test:
    pred = model.predict(test_ans[0])
    data.append(abs(pred - test_ans[1])/test_ans[1])
    results.append(f'{float(pred)}  {float(test_ans[1])}')
    MSE.append((pred-test_ans[1])**2)
    MAPE.append(abs((pred-test_ans[1])/test_ans[1]))

MSE = np.sum(MSE)/len(MSE)
MAPE = 100*np.sum(MAPE)/len(MAPE)

print(f"MSE : {MSE}")
print(f"MAPE : {MAPE}")
if train:
    print(f"time : {t_end - t_start}")
    with open(f'../result/{data_name}.txt','w') as f:
        for r in results:
            f.write(f'{r}\n')
            print(r)
        f.write(f'MSE : {MSE}\n')
        f.write(f'MAPE : {MAPE}\n')
        f.write(f'time : {t_end - t_start}\n')
else:
    for r in results:
        print(r)
