import os
import random
import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt

# --- НАСТРОЙКИ ---
# Путь к обученным весам (появятся после обучения)
MODEL_PATH = 'runs/detect/runs/detect/plate_detector/weights/best.pt'
# Путь к папке с тестовыми изображениями
TEST_IMAGES_PATH = 'Dataset2/test/images'


def run_test():
    # 1. Загрузка модели
    if not os.path.exists(MODEL_PATH):
        print(f"Ошибка: Веса {MODEL_PATH} не найдены. Сначала закончи обучение!")
        return

    model = YOLO(MODEL_PATH)

    # 2. Выбор случайного изображения
    images = [f for f in os.listdir(TEST_IMAGES_PATH) if f.endswith(('.png', '.jpg', '.jpeg'))]
    if not images:
        print("В папке test/images нет картинок!")
        return

    random_img = random.choice(images)
    img_path = os.path.join(TEST_IMAGES_PATH, random_img)

    # 3. Предсказание
    results = model.predict(source=img_path, conf=0.25, save=False)

    # 4. Визуализация
    # YOLOv8 умеет сама рисовать результат через метод .plot()
    res_plotted = results[0].plot()

    # Конвертируем из BGR (OpenCV) в RGB (Matplotlib)
    res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(10, 6))
    plt.imshow(res_rgb)
    plt.title(f"Детекция на файле: {random_img}")
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    run_test()