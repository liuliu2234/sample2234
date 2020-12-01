import os, shutil
original_dataset_dir = r'/Users/capybara/Downloads/dogs-vs-cats/train'

base_dir = r'/Users/capybara/Downloads/cats_and_dogs_small' # 儲存少量資料集的目錄位置
if not os.path.isdir(base_dir): os.mkdir(base_dir) # 如果目錄不在 才建立目錄

train_dir = os.path.join(base_dir, 'train') # 訓練目錄位置
if not os.path.isdir(train_dir): os.mkdir(train_dir)

validation_dir = os.path.join(base_dir, 'validation') # 驗證目錄位置
if not os.path.isdir(validation_dir): os.mkdir(validation_dir)

test_dir = os.path.join(base_dir, 'test') # 測試目錄位置
if not os.path.isdir(test_dir): os.mkdir(test_dir)

train_cats_dir = os.path.join(train_dir, 'cats') # 用來訓練的貓圖片目錄
if not os.path.isdir(train_cats_dir): os.mkdir(train_cats_dir)

train_dogs_dir = os.path.join(train_dir, 'dogs') # 用來訓練的狗圖片目錄
if not os.path.isdir(train_dogs_dir): os.mkdir(train_dogs_dir)

validation_cats_dir = os.path.join(validation_dir, 'cats') # 用來驗證的貓圖片目錄
if not os.path.isdir(validation_cats_dir): os.mkdir(validation_cats_dir)

validation_dogs_dir = os.path.join(validation_dir, 'dogs') # 用來驗證的狗圖片目錄
if not os.path.isdir(validation_dogs_dir): os.mkdir(validation_dogs_dir)

test_cats_dir = os.path.join(test_dir, 'cats') # 用來測試的貓圖片目錄
if not os.path.isdir(test_cats_dir): os.mkdir(test_cats_dir)

test_dogs_dir = os.path.join(test_dir, 'dogs') # 用來測試的狗圖片目錄
if not os.path.isdir(test_dogs_dir): os.mkdir(test_dogs_dir)

# 複製前面1000張貓圖片到train_cats_dir訓練目錄
fnames = ['cat.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_cats_dir, fname)
    shutil.copyfile(src, dst)

# 複製下500張貓圖片到validation_cats_dir驗證目錄
fnames = ['cat.{}.jpg'.format(i) for i in range(1000, 1500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validation_cats_dir, fname)
    shutil.copyfile(src, dst)

# 複製下500張貓圖片到test_cats_dir測試目錄
fnames = ['cat.{}.jpg'.format(i) for i in range(1500, 2000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(test_cats_dir, fname)
    shutil.copyfile(src, dst)

# 複製前面1000張狗圖片到train_cats_dir訓練目錄
fnames = ['dog.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_dogs_dir, fname)
    shutil.copyfile(src, dst)

# 複製下500張狗圖片到validation_cats_dir驗證目錄
fnames = ['dog.{}.jpg'.format(i) for i in range(1000, 1500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validation_dogs_dir, fname)
    shutil.copyfile(src, dst)

# 複製下500張狗圖片到test_cats_dir測試目錄
fnames = ['dog.{}.jpg'.format(i) for i in range(1500, 2000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(test_dogs_dir, fname)
    shutil.copyfile(src, dst)

# ======================================================
# 資料擴增法,
# 透過產生器ImageDataGenerator設定資料擴增
from keras.preprocessing.image import ImageDataGenerator

'''
datagen = ImageDataGenerator(rotation_range=40, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')

# 顯示一些隨機擴充的訓練影像
import matplotlib.pyplot as plt
from keras.preprocessing import image

fnames = [os.path.join(train_cats_dir, fname) for fname in os.listdir(train_cats_dir)]

img_path = fnames[3] # 選擇一張影像來擴充
img = image.load_img(img_path, target_size=(150, 150)) # 讀取影像並調整大小
x = image.img_to_array(img)
x = x.reshape((1, ) + x.shape)

i = 0
for batch in datagen.flow(x, batch_size=1):
    plt.figure(i)
    imgplot = plt.imshow(image.array_to_img(batch[0]))
    i += 1
    if i % 4 == 0:
        break
plt.show()
'''

# 定義具有Dropout層的新卷積神經網路（在展平層後方加入Dropout層）
from keras import layers
from keras import models
from keras import optimizers

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5)) # 丟棄50%
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4), metrics=['acc'])

# 使用資料擴增產生器來訓練卷積神經網路
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=40, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255) # 驗證資料不擴充

train_generator = train_datagen.flow_from_directory(train_dir, target_size=(150, 150), batch_size=16, class_mode='binary')
validation_generator = test_datagen.flow_from_directory(validation_dir, target_size=(150, 150), batch_size=16, class_mode='binary')

history = model.fit_generator(train_generator, steps_per_epoch=round(len(train_generator)/16), epochs=100, validation_data=validation_generator, validation_steps=30)
#steps_per_epoch=round(len(train_generator)/batch_size)
model.save('cats_and_dogs_small_2.h5')

# 顯示訓練和驗證週期的損失值和準確度曲線
import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and Validation accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and Validation loss')
plt.legend()

plt.show()

# 沒有足夠的 memory/RAM 儲存batch資料 嘗試降低batch_size, validation_steps