import cv2
import glob
import numpy as np
import math
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from scipy import stats

img_path_pos = glob.glob("D:/winuser/Desktop/HW3/dataset/pos/*.jpg")
img_path_neg = glob.glob("D:/winuser/Desktop/HW3/dataset/neg/*.jpg")
pos_mean_matrix = np.zeros((10, 10), dtype=np.float)
neg_mean_matrix = np.zeros((10, 10), dtype=np.float)

for path1 in img_path_pos:
    img_pos = cv2.imread(path1)
    img_pos_resize = cv2.resize(img_pos, (10, 10))
    img_pos_gray = cv2.cvtColor(img_pos_resize, cv2.COLOR_BGR2GRAY)
    pos_mean_matrix += img_pos_gray / len(img_path_pos)
cv2.imwrite("D:/winuser/Desktop/DM and Vis/HW3/pos_mean_matrix.jpg", pos_mean_matrix)
for path1 in img_path_neg:
    img_neg = cv2.imread(path1)
    img_neg_resize = cv2.resize(img_neg, (10, 10))
    img_neg_gray = cv2.cvtColor(img_neg_resize, cv2.COLOR_BGR2GRAY)
    neg_mean_matrix += img_neg_gray / len(img_path_neg)

plt.hist(pos_mean_matrix.flatten(), normed=True, color='pink', edgecolor='k')
plt.hist(neg_mean_matrix.flatten(), normed=True, color='lightblue', edgecolor='k')

line1 = sns.kdeplot(pos_mean_matrix.flatten(), color='r', linewidth=2, label='pos')
line2 = sns.kdeplot(neg_mean_matrix.flatten(), color='b', linewidth=2, label='neg')
plt.tick_params(top='off', right='off')
plt.xlabel('Brightness distribution(0 ~ 255)')
plt.ylabel('Probability')
plt.legend([line1, line2], labels=['pos', 'neg'], loc='best')
plt.savefig('D:/winuser/Desktop/DM and Vis/HW3/pic.jpg')
plt.close('all')

gaussian1 = stats.norm(loc=pos_mean_matrix.mean(), scale=1.0)
gaussian2 = stats.norm(loc=neg_mean_matrix.mean(), scale=1.0)
x2 = np.linspace(113.0, 140.0, 1000)
y1 = gaussian1.pdf(x2)
y2 = gaussian2.pdf(x2)
plt.xlabel('Brightness distribution(0 ~ 255)')
plt.ylabel('Probability')
a1 = plt.plot(x2, y1, color='red', label="pos")
a2 = plt.plot(x2, y2, color='blue', label='neg')
plt.legend([a1, a2], labels=['pos', 'neg'], loc='best')
plt.savefig('D:/winuser/Desktop/DM and Vis/HW3/gaussian.jpg')

for k in range(2):
    test = cv2.imread("D:/winuser/Desktop/DM and Vis/HW3/test{}.jpg".format(k + 1))
    test_gray = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)
    test_h = test_gray.shape[0]
    test_w = test_gray.shape[1]
    for i in range(math.floor(test_h / 10) - 1):
        for j in range(math.floor(test_w / 10) - 1):
            crop_test = test_gray[i * 10: (i + 1) * 10, j * 10: (j + 1) * 10]
            pixel = 0
            for m in range(10):
                for n in range(10):
                    pixel += crop_test[m, n]
            pixel_mean = pixel / 100
            if pixel_mean >= (pos_mean_matrix.mean() + neg_mean_matrix.mean()) / 2:
                cv2.rectangle(test, (j * 10, i * 10), ((j + 1) * 10, (i + 1) * 10), (0, 255, 0), 2)
        cv2.imwrite("D:/winuser/Desktop/DM and Vis/HW3/test{}_result1.jpg".format(k + 1), test)
