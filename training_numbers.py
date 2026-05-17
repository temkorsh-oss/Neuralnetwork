import os
import cv2
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# --- 1. ТЕХНИЧЕСКИЕ НАСТРОЙКИ ---
# Размеры входных изображений для нейросети
IMG_WIDTH = 128
IMG_HEIGHT = 64
# Максимальное количество символов в номерном знаке РФ
MAX_CHARS = 9
# Список разрешенных символов (цифры и буквы РФ согласно ГОСТ)
CHAR_LIST = "0123456789ABEKMHOPCTYX"
# Словарь для преобразования символов в числовые индексы
char_to_num = {char: i for i, char in enumerate(CHAR_LIST)}
# Общее число классов (+1 для обозначения пустого пространства)
NUM_CLASSES = len(CHAR_LIST) + 1


def encode_label(label):
    """
    Преобразует текстовую строку в числовой вектор фиксированной длины.
    """
    encoded = [char_to_num[c] for c in label if c in char_to_num]
    # Дополнение вектора значением "пустоты" до достижения MAX_CHARS
    while len(encoded) < MAX_CHARS:
        encoded.append(len(CHAR_LIST))
    return encoded[:MAX_CHARS]


def load_data(base_path):
    """
    Загружает изображения и аннотации из указанной директории.
    """
    images, labels = [], []
    ann_dir = os.path.join(base_path, 'ann')
    img_dir = os.path.join(base_path, 'img')

    if not os.path.exists(ann_dir):
        print(f"Директория {ann_dir} не найдена.")
        return np.array([]), np.array([])

    # Получение списка всех файлов разметки
    files = [f for f in os.listdir(ann_dir) if f.endswith('.json')]
    print(f"Загрузка данных из {base_path}: обнаружено {len(files)} файлов.")

    for i, filename in enumerate(files):
        # Чтение текстовой метки из JSON
        with open(os.path.join(ann_dir, filename), 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Подготовка текста (верхний регистр, удаление пробелов)
        label_text = data['description'].upper().replace(" ", "")
        base_name = os.path.splitext(filename)[0]
        img_path = os.path.join(img_dir, base_name + '.png')

        if os.path.exists(img_path):
            # Загрузка изображения в режиме оттенков серого (Grayscale)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                # Масштабирование до размеров 128x64
                img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
                images.append(img)
                labels.append(encode_label(label_text))

        # Вывод статуса загрузки каждые 1000 изображений
        if i > 0 and i % 1000 == 0:
            print(f"Обработано {i} из {len(files)}...")

    return np.array(images), np.array(labels)


# --- 2. АРХИТЕКТУРА НЕЙРОННОЙ СЕТИ ---
def build_model():
    """
    Создает структуру сверточной нейронной сети для распознавания текста.
    """
    model = models.Sequential([
        # Входной слой (64x128 пикселей, 1 канал цвета)
        layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 1)),

        # Первый блок свертки: поиск базовых геометрических признаков
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),

        # Второй блок свертки: извлечение более сложных форм
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),

        # Третий блок свертки: анализ высокоуровневых паттернов
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),

        # Преобразование данных в одномерный вектор
        layers.Flatten(),

        # Полносвязный слой для глубокого анализа признаков
        layers.Dense(512, activation='relu'),
        # Dropout для предотвращения переобучения (случайное отключение 40% нейронов)
        layers.Dropout(0.4),

        # Выходной слой: предсказание вероятностей для 9 позиций
        layers.Dense(MAX_CHARS * NUM_CLASSES, activation='softmax'),
        # Преобразование выхода в матрицу (9 позиций на количество символов)
        layers.Reshape((MAX_CHARS, NUM_CLASSES))
    ])

    # Настройка процесса обучения (оптимизатор Adam, функция потерь категориальная кросс-энтропия)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


# --- 3. ПОДГОТОВКА И ЗАПУСК ОБУЧЕНИЯ ---
if __name__ == "__main__":
    # Загрузка данных из Dataset1 (train и val)
    X_train, y_train = load_data('Dataset1/train')
    X_val, y_val = load_data('Dataset1/val')

    if len(X_train) > 0:
        # Добавление оси канала для совместимости с Conv2D слоями
        X_train = np.expand_dims(X_train, axis=-1)
        X_val = np.expand_dims(X_val, axis=-1)

        # Конфигурация аугментации данных (искажения для повышения устойчивости модели)
        datagen = ImageDataGenerator(
            rescale=1. / 255,  # Нормализация значений пикселей в диапазон [0, 1]
            rotation_range=12,  # Случайный поворот (до 12 градусов)
            width_shift_range=0.1,  # Горизонтальное смещение
            height_shift_range=0.1,  # Вертикальное смещение
            shear_range=0.2,  # Сдвиг (важно для корректного чтения под углом)
            zoom_range=0.1,  # Случайное масштабирование
            fill_mode='nearest'  # Заполнение пустых областей ближайшими пикселями
        )

        # Для валидации используем только нормализацию (без искажений)
        val_datagen = ImageDataGenerator(rescale=1. / 255)

        model = build_model()
        model.summary()  # Вывод структуры модели в консоль

        print("\n Инициализация процесса обучения...")
        batch_size = 32

        # Запуск обучения на 25 эпохах
        history = model.fit(
            datagen.flow(X_train, y_train, batch_size=batch_size),
            validation_data=val_datagen.flow(X_val, y_val, batch_size=batch_size),
            epochs=25,
        )

        # Сохранение обученной модели
        model.save('ocr_license_plate_model_v1.keras')

        # Сохранение статистики обучения в CSV файл (Pandas)
        pd.DataFrame(history.history).to_csv('training_stats.csv', index=False)

        print("\nПроцесс завершен. Модель и статистика сохранены.")
    else:
        print("Ошибка: тренировочные данные не загружены. Проверьте путь Dataset1/train")