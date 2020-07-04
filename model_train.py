import numpy as np
from sklearn import preprocessing
from tensorflow_core.python.keras.models import Sequential
from tensorflow_core.python.keras.layers.core import Dense
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from tensorflow_core.python.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import pre_processing


def create_model():
    checkpoint = ModelCheckpoint('sdr_model.h5', monitor='accuracy', verbose=1,
                                 save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    gray_data = np.load("npy_data/gray_dataset.npy")
    color_data = np.load("npy_data/color_dataset.npy")
    # img_pixel_dataset = np.load("npy_data/img_pixel_dataset.npy")
    label = np.load("npy_data/label.npy")


    # dataset = pre_processing.npy_dataset_concatenate(gray_data, color_data)
    dataset = pre_processing.npy_dataset_concatenate(gray_data, color_data)
    # corr_matrix = np.corrcoef(dataset)
    # print(corr_matrix)
    le = preprocessing.LabelEncoder()
    label = le.fit_transform(label)

    x_train, x_test, y_train, y_test = train_test_split(dataset, label, test_size=0.20, shuffle=True)

    model = Sequential()
    model.add(Dense(14, input_dim=14, activation=None))
    model.add(Dense(128, activation='tanh'))
    model.add(Dense(256, activation='sigmoid'))
    model.add(Dense(3, activation='softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=150, verbose=0, batch_size=20, shuffle=True, callbacks=callbacks_list)

    pred_y_test = model.predict_classes(x_test)

    acc_model = accuracy_score(y_test, pred_y_test)
    print("Prediction Acc model:", acc_model)
    print("Org. Labels:", y_test[:30])
    print("Pred Labels:", (pred_y_test[:30]))
    # c_report = classification_report(y_test, pred_y_test, zero_division=0)
    # print(c_report)
    print("\n\n")
