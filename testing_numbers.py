import os
import cv2
import json
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image

# --- НАСТРОЙКИ ---
IMG_WIDTH = 128
IMG_HEIGHT = 64
CHAR_LIST = "0123456789ABEKMHOPCTYX"
index_to_char = {i: char for i, char in enumerate(CHAR_LIST)}


def decode_prediction(pred_array):
    """Преобразует выход нейросети в строку текста."""
    result = ""
    for char_probs in pred_array:
        char_index = np.argmax(char_probs)
        if char_index < len(CHAR_LIST):
            result += index_to_char[char_index]
    return result


def show_random_test(model_path, test_dir):
    # 1. Загрузка модели
    if not os.path.exists(model_path):
        print("Файл модели не найден!")
        return
    model = tf.keras.models.load_model(model_path)

    # 2. Выбор случайного файла
    ann_dir = os.path.join(test_dir, 'ann')
    img_dir = os.path.join(test_dir, 'img')

    all_json_files = [f for f in os.listdir(ann_dir) if f.endswith('.json')]
    random_json = random.choice(all_json_files)

    # 3. Получение правильного ответа
    with open(os.path.join(ann_dir, random_json), 'r', encoding='utf-8') as f:
        data = json.load(f)
    true_text = data['description'].upper().replace(" ", "")

    # 4. Загрузка и подготовка изображения
    img_name = os.path.splitext(random_json)[0] + '.png'
    img_path = os.path.join(img_dir, img_name)

    # Читаем оригинал для показа в окне
    original_img = cv2.imread(img_path)
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

    # Подготовка для нейросети
    gray_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    resized_img = cv2.resize(gray_img, (IMG_WIDTH, IMG_HEIGHT))
    normalized_img = resized_img.astype('float32') / 255.0
    input_data = np.expand_dims(np.expand_dims(normalized_img, axis=-1), axis=0)

    # 5. Предсказание
    prediction = model.predict(input_data, verbose=0)
    predicted_text = decode_prediction(prediction[0])

    # 6. Вывод результата в консоль и на экран
    print(f"\nРезультат проверки:")
    print(f"Файл: {img_name}")
    print(f"Реальный номер:    {true_text}")
    print(f"Распознанный номер: {predicted_text}")

    color = 'green' if true_text == predicted_text else 'red'

    plt.figure(figsize=(8, 4))
    plt.imshow(original_img)
    plt.title(f"True: {true_text}\nPred: {predicted_text}", color=color, fontsize=16)
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    # Укажи путь к своей модели и папке test
    show_random_test('ocr_license_plate_model_v1.h5', 'Dataset1/test')