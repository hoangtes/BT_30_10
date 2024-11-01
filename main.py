import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB

# Đường dẫn tới thư mục chứa ảnh nha khoa
data_path = 'anh/'  # Thay đổi đường dẫn nếu cần
image_size = (64, 64)  # Kích thước ảnh đầu vào
X_images = []
y_images = []  # Nhãn tương ứng với từng ảnh

# Đọc và tiền xử lý ảnh
for label in os.listdir(data_path):
    label_path = os.path.join(data_path, label)
    if os.path.isdir(label_path):  # Kiểm tra xem có phải là thư mục không
        for image_file in os.listdir(label_path):
            img_path = os.path.join(label_path, image_file)
            img = cv2.imread(img_path)
            img = cv2.resize(img, image_size)  # Thay đổi kích thước ảnh
            X_images.append(img)
            y_images.append(label)

# Chuyển đổi danh sách thành mảng NumPy
X_images = np.array(X_images)
y_images = np.array(y_images)

# Chuyển đổi nhãn thành số
label_encoder = LabelEncoder()
y_images_encoded = label_encoder.fit_transform(y_images)

# Chia dữ liệu thành tập huấn luyện và kiểm tra
X_train_images, X_test_images, y_train_images, y_test_images = train_test_split(X_images, y_images_encoded, test_size=0.2, random_state=42)

# Chuyển đổi dữ liệu hình ảnh thành dạng một chiều
X_train_images_flat = X_train_images.reshape(X_train_images.shape[0], -1)
X_test_images_flat = X_test_images.reshape(X_test_images.shape[0], -1)

# Naive Bayes
nb_model = GaussianNB()
nb_model.fit(X_train_images_flat, y_train_images)
y_pred_nb = nb_model.predict(X_test_images_flat)
print("Accuracy of Naive Bayes on dental images:", accuracy_score(y_test_images, y_pred_nb))

# CART (Gini Index)
cart_model_images = DecisionTreeClassifier(criterion='gini')
cart_model_images.fit(X_train_images_flat, y_train_images)
y_pred_cart_images = cart_model_images.predict(X_test_images_flat)
print("Accuracy of CART on dental images:", accuracy_score(y_test_images, y_pred_cart_images))

# ID3 (Information Gain)
id3_model_images = DecisionTreeClassifier(criterion='entropy')
id3_model_images.fit(X_train_images_flat, y_train_images)
y_pred_id3_images = id3_model_images.predict(X_test_images_flat)
print("Accuracy of ID3 on dental images:", accuracy_score(y_test_images, y_pred_id3_images))

# Neural Network
nn_model_images = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000)
nn_model_images.fit(X_train_images_flat, y_train_images)
y_pred_nn_images = nn_model_images.predict(X_test_images_flat)
print("Accuracy of Neural Network on dental images:", accuracy_score(y_test_images, y_pred_nn_images))
