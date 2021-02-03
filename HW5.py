import matplotlib.pyplot as plt
import glob
import cv2
import numpy as np
from skimage import feature, data, color, transform


img_path_pos = glob.glob("D:/winuser/Desktop/HW3/dataset/pos/*.jpg")
img_path_neg = glob.glob("D:/winuser/Desktop/HW3/dataset/neg/*.jpg")
matrix = np.zeros([400, 50, 50], dtype=np.float)
i = 0

for path1 in img_path_pos:
    img_pos = cv2.imread(path1)
    img_pos_resize = cv2.resize(img_pos, (50, 50))
    img_pos_gray = cv2.cvtColor(img_pos_resize, cv2.COLOR_BGR2GRAY)
    matrix[i] = img_pos_gray
    i += 1

for path1 in img_path_neg:
    img_neg = cv2.imread(path1)
    img_neg_resize = cv2.resize(img_neg, (50, 50))
    img_neg_gray = cv2.cvtColor(img_neg_resize, cv2.COLOR_BGR2GRAY)
    matrix[i] = img_neg_gray
    i += 1

X_train = np.array([feature.hog(im) for im in matrix])
y_train = np.zeros(X_train.shape[0])
y_train[:200] = 1
print(X_train[0].shape)

from sklearn.naive_bayes import GaussianNB
from sklearn.cross_validation import cross_val_score

cross_val_score(GaussianNB(), X_train, y_train)
print(cross_val_score(GaussianNB(), X_train, y_train))

from sklearn.svm import LinearSVC
from sklearn.grid_search import GridSearchCV
grid = GridSearchCV(LinearSVC(), {'C': [1.0, 2.0, 4.0, 8.0]})
grid.fit(X_train, y_train)
print(grid.best_score_)
print(grid.best_params_)

model = grid.best_estimator_
model.fit(X_train, y_train)

test_image = data.astronaut()
test_image = color.rgb2gray(test_image)
test_image = transform.rescale(test_image, 0.5)
test_image = test_image[:160, 40:180]
plt.imshow(test_image, cmap='gray')
plt.axis('off')
plt.show()

def sliding_window(img, patch_size=matrix[0].shape, istep=2, jstep=2, scale=1.0):
    Ni, Nj = (int(scale * s) for s in patch_size)
    for i in range(0, img.shape[0] - Ni, istep):
        for j in range(0, img.shape[1] - Ni, jstep):
            patch = img[i:i + Ni, j:j + Nj]
            if scale != 1:
                patch = cv2.resize(patch, patch_size)
            yield (i, j), patch

indices, patches = zip(*sliding_window(test_image))
patches_hog = np.array([feature.hog(patch) for patch in patches])
print(patches_hog.shape)

labels = model.predict(patches_hog)
print(labels.sum())

fig, ax = plt.subplots()
ax.imshow(test_image, cmap='gray')
ax.axis('off')

Ni, Nj = matrix[0].shape
indices = np.array(indices)

for i, j in indices[labels == 1]:
    ax.add_patch(plt.Rectangle((j, i), Nj, Ni, edgecolor='red',
                               alpha=0.3, lw=2, facecolor='none'))
plt.savefig("D:/winuser/Desktop/DM and Vis/HW5/my_part (50, 50).jpg")

'''hog_vec, hog_vis = feature.hog(img_pos_gray, visualize=True)
    fig, ax = plt.subplots(1, 2, figsize=(12, 6),
                           subplot_kw=dict(xticks=[], yticks=[]))
    ax[0].imshow(img_pos_gray, cmap='gray')
    ax[0].set_title('input image')

    ax[1].imshow(hog_vis)
    ax[1].set_title('visualization of HOG features')
    plt.show()'''