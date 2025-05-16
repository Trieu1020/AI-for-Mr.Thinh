import cv2
import numpy as np
import os
from ultralytics import YOLO
from tensorflow.keras.models import load_model
import time
import matplotlib.pyplot as plt

# --- Cấu hình ---
image_size = 200
model_path = "E:/AI for Mr.Thinh/best_model.keras"
folder_path = "E:/AI for Mr.Thinh/cropped_food"

# --- Danh sách món ăn ---
dishes_classes = [
    'Braised fish', 'Braised meat', 'Braised meat w egg',
    'Fried chicken', 'Fried egg', 'Mustard soup',
    'Rice', 'Sour Broth', 'Spinach', 'Tofu'
]

# --- Bảng giá ---
price_dict = {
    'Braised fish': 25000,
    'Braised meat': 30000,
    'Braised meat w egg': 32000,
    'Fried chicken': 35000,
    'Fried egg': 10000,
    'Mustard soup': 12000,
    'Rice': 5000,
    'Sour Broth': 15000,
    'Spinach': 10000,
    'Tofu': 8000
}

# --- Load mô hình ---
model = load_model(model_path)

# --- Hàm xử lý ảnh đã cắt: nhận diện và tính tiền ---
def recognize_and_display(bowl_paths):
    total_price = 0
    recognized_dishes = {}

    for image_path in bowl_paths:
        try:
            img = cv2.imread(image_path)
            img_resized = cv2.resize(img, (image_size, image_size))
            img_normalized = img_resized / 255.0
            img_input = np.expand_dims(img_normalized, axis=0)

            predictions = model.predict(img_input)[0]
            class_index = np.argmax(predictions)
            dish_name = dishes_classes[class_index]
            confidence = predictions[class_index]
            price = price_dict[dish_name]
            total_price += price
            recognized_dishes[dish_name] = recognized_dishes.get(dish_name, 0) + 1

            label_text = f"{dish_name} - {price} VND"
            cv2.putText(img, label_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.imshow(img_rgb)
            plt.title(os.path.basename(image_path))
            plt.axis('off')
            plt.show()

        except Exception as e:
            print(f"❌ Lỗi với ảnh {image_path}: {e}")

    print("\n🧾 TỔNG KẾT:")
    for dish, count in recognized_dishes.items():
        price = price_dict[dish]
        subtotal = price * count
        print(f"- {dish}: {count} món × {price} = {subtotal} VND")
    print(f"💰 Tổng cộng: {total_price:,} VND")

# --- Hàm cắt bowl từ ảnh ---
def crop_bowls_from_image(image_path, output_folder, confidence_threshold=0.1):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    model_yolo = YOLO("yolov8n.pt")
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Không thể đọc ảnh từ {image_path}")
        return []

    results = model_yolo(img, conf=confidence_threshold)
    detections = results[0]
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    bowl_paths = []

    for i, detection in enumerate(detections.boxes.data.tolist()):
        x1, y1, x2, y2, confidence, class_id = detection
        class_name = detections.names[int(class_id)]

        if class_name != "bowl":
            continue

        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cropped_object = img[y1:y2, x1:x2]
        if cropped_object.size == 0:
            continue

        output_path = os.path.join(output_folder, f"{base_name}_bowl_{i}.jpg")
        cv2.imwrite(output_path, cropped_object)
        print(f"📂 Đã lưu bowl ({confidence:.2f}) → {output_path}")
        bowl_paths.append(output_path)

    print(f"✅ Hoàn tất! Đã cắt {len(bowl_paths)} bowl từ ảnh.")
    return bowl_paths

# --- Chạy camera ---
def run_camera_and_crop_bowls():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Không thể mở camera")
        return

    print("🎥 Đang chạy camera... Nhấn C để chụp & nhận diện, O để thoát")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Không thể đọc từ camera")
            break

        cv2.imshow("Camera - Nhấn C để chụp, O để thoát", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):
            timestamp = int(time.time())
            image_path = f"tray_{timestamp}.jpg"
            cv2.imwrite(image_path, frame)
            print(f"📸 Đã chụp ảnh → {image_path}")

            bowl_paths = crop_bowls_from_image(image_path, folder_path, confidence_threshold=0.1)
            recognize_and_display(bowl_paths)

        elif key == ord('o'):
            print("👋 Thoát chương trình.")
            break

    cap.release()
    cv2.destroyAllWindows()

# --- Run chương trình ---
if __name__ == "__main__":
    run_camera_and_crop_bowls()
