# Keras迴歸範例,監督式學習,使用boston_housing資料集預測房價
from keras.datasets import boston_housing
(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

# 正規化資料
mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std

test_data -= mean
test_data /= std

# 模型定義
from keras import models
from keras import layers

def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(train_data.shape[1], )))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1)) # 線性層
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model

# K-fold 驗證
import numpy as np

k = 4
num_val_samples = len(train_data) // k
"""
num_epochs = 100
all_scores = []

for i in range(k):
    print('processing fold #', i)
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
    partial_train_data = np.concatenate([train_data[:i * num_val_samples], train_data[(i + 1) * num_val_samples:]], axis=0)
    partial_train_targets = np.concatenate([train_targets[:i * num_val_samples], train_targets[(i + 1) * num_val_samples:]], axis=0)

    model = build_model()
    model.fit(partial_train_data, partial_train_targets, epochs=num_epochs, batch_size=1, verbose=0)

    val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
    all_scores.append(val_mae)

print(all_scores)
print(np.mean(all_scores))
"""
#======================================
# 儲存每折的驗證紀錄
num_epochs = 500
all_mae_histories = []
for i in range(k):
    print('processing fold #', i)
    val_data = train_data[i * num_val_samples:(i + 1) * num_val_samples] # 驗證資料來自 k 區塊
    val_targets = train_targets[i * num_val_samples:(i + 1) * num_val_samples]

    partial_train_data = np.concatenate([train_data[:i * num_val_samples], train_data[(i + 1) * num_val_samples:]], axis=0)
    # 訓練資料來自 k 以外的所有區塊
    partial_train_targets = np.concatenate([train_targets[:i * num_val_samples], train_targets[(i + 1) * num_val_samples:]], axis=0)

    model = build_model()
    history = model.fit(partial_train_data, partial_train_targets, validation_data=(val_data, val_targets), epochs=num_epochs, batch_size=1, verbose=0)
    mae_history = history.history['val_mean_absolute_error']
    all_mae_histories.append(mae_history)

# 建立連續平均K折驗證分數的歷史
average_mae_history = [np.mean([x[i] for x in all_mae_histories]for i in range(num_epochs))]

# 繪製驗證分數
import matplotlib.pyplot as plt

plt.plot(range(1, len(average_mae_history)+ 1), average_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()

#======================================
# 排除前10個資料點,繪製驗證分數

def smooth_curve(points, factor=0.9): # 指數值為0.9
    smoothed_points = []
    for point in points:
        if smoothed_points:
            prenious = smoothed_points[-1]
            smoothed_points.append(prenious * factor + point *(1 - factor)) # 指數平均數
        else:
            smoothed_points.append(point)
    return smoothed_points

smooth_mae_history = smooth_curve(average_mae_history[10:]) # 先去除前10個資料再平均

plt.plot(range(1, len(smooth_mae_history)+ 1), smooth_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()
# 上圖可看出MAE在80個週期後開始過度調配

# 訓練最終模型
model = build_model() # 建立一個用最佳參數compile過的模型
model.fit(train_data, train_targets, epochs=80, batch_size=16, verbose=0) # 以整個資料進行訓練
test_mse_score, test_mae_score = model.evaluate(test_data, test_targets)

print(test_mae_score)