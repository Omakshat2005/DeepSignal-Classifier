#Rf_modulator_detector
import matplotlib.pyplot as plt
import pickle
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import gc
# Use this EXACT list in both files
with open('Rml.pkl', 'rb') as f:
    data = pickle.load(f, encoding='latin1')
classes = ['8PSK', 'AM-DSB', 'AM-SSB', 'BPSK', 'CPFSK', 'GFSK', 'PAM4', 'QAM16', 'QAM64', 'QPSK', 'WBFM']

mod_to_id = {name: i for i, name in enumerate(classes)}
cleaned_data={(mod_to_id[k[0]],k[1]):v for k, v in data.items()}
X = []
lbl = []
for k, v in data.items():
    X.append(v)
    for i in range(v.shape[0]):
        lbl.append(k)
X = np.vstack(X) 
X = np.transpose(X, (0, 2, 1)).astype('float32')
mods_labels = np.array([mod_to_id[l[0]] for l in lbl])
snr_labels = np.array([l[1] for l in lbl])
print("Signals Shape (X):", X.shape)    
print("Target Labels (Y):", mods_labels.shape) 
Y = to_categorical(mods_labels)
X_train, X_test, Y_train, Y_test, snr_train, snr_test = train_test_split(
    X, Y, snr_labels, test_size=0.2, random_state=42)
print(f"Training shapes: X={X_train.shape}, Y={Y_train.shape}")
print(f"Testing shapes: X={X_test.shape}, Y={Y_test.shape}")
del X, Y, mods_labels  
gc.collect()
num_classes = Y_train.shape[1] 

model = Sequential([
    LSTM(64, input_shape=(128, 2), return_sequences=True),
    BatchNormalization(), 
    Dropout(0.2),         
    Dropout(0.2),
    Dense(128, activation='relu'),
    Dense(num_classes, activation='softmax')
])

model.compile(loss='categorical_crossentropy', 
              optimizer='adam', 
              metrics=['accuracy'])

model.summary()
print("Starting Training...")
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=0.00001)

history = model.fit(
    X_train, Y_train,
    epochs=25,
    batch_size=1024,
    validation_data=(X_test, Y_test),
    callbacks=[early_stop, reduce_lr],
    verbose=1
)
model.save('mod_classifier_lstm.h5')
print("Model Saved!")