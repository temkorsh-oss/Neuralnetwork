import os
import torch
from ultralytics import YOLO

# --- ПУТИ И НАСТРОЙКИ ---
PROJECT_ROOT = os.getcwd()
CONFIG_PATH = os.path.join(PROJECT_ROOT, 'Dataset2', 'data.yaml')


def train_plate_detector():
    """
    Запускает обучение модели YOLOv8 для детекции номеров.
    Оптимизировано для использования видеокарты (GPU).
    """
    # 1. Проверка доступности GPU (CUDA)
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        print(f"CUDA активирована. Используем видеокарту: {device_name}")
    else:
        print("Внимание: Видеокарта не обнаружена! PyTorch не видит CUDA.")
        print("Обучение переключится на процессор, что займет много времени.")
        return

    # 2. Проверка файла конфигурации
    if not os.path.exists(CONFIG_PATH):
        print(f"Ошибка: Файл конфигурации не найден по пути {CONFIG_PATH}")
        return

    print("\nЗагрузка архитектуры YOLOv8 Nano...")
    model = YOLO('yolov8n.pt')

    print("Старт обучения...\n")

    try:
        model.train(
            data=CONFIG_PATH,
            epochs=30,
            imgsz=640,
            batch=32,  # Агрессивная загрузка видеопамяти
            device=0,  # Использование основной GPU
            workers=8,  # 8 потоков для подгрузки данных
            project='runs/detect',
            name='plate_detector',
            exist_ok=True,
            pretrained=True,
            amp=True,  # Смешанная точность (буст скорости)
            verbose=True
        )
        print("\n Обучение успешно завершено!")
        print("Сохраненные веса лежат здесь: runs/detect/plate_detector/weights/best.pt")

    except Exception as e:
        print(f"\n Произошла ошибка во время обучения: {e}")


if __name__ == '__main__':
    train_plate_detector()