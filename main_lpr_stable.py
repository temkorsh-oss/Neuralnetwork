import sys
import os
import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO

# ==================== ФИКС ОШИБКИ QT ПЛАГИНОВ ====================
from PyQt5 import QtWidgets

qt_plugin_path = os.path.join(os.path.dirname(QtWidgets.__file__), 'Qt5', 'plugins')
if os.path.exists(qt_plugin_path):
    os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = qt_plugin_path

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QProgressBar, QMessageBox, QFrame
)
from PyQt5.QtGui import QPixmap, QImage, QIcon
from PyQt5.QtCore import Qt, QThread, pyqtSignal

# ==================== НАСТРОЙКИ И ПУТИ К МОДЕЛЯМ ====================
DETECTOR_PATH = 'runs/detect/runs/detect/plate_detector/weights/best.pt'
READER_PATH = 'ocr_license_plate_model_v1.h5'

IMG_WIDTH, IMG_HEIGHT = 128, 64
CHAR_LIST = "0123456789ABEKMHOPCTYX"
CONFIDENCE_THRESHOLD = 0.45

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# ==================== ПОТОК ВЫЧИСЛЕНИЙ (LOGIC) ===================
class LPRThread(QThread):
    """Класс для выполнения тяжелых расчетов в фоновом режиме"""
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, detector, reader, image_path):
        super().__init__()
        self.detector = detector
        self.reader = reader
        self.image_path = image_path
        self.num_to_char = {i: char for i, char in enumerate(CHAR_LIST)}

    def decode_prediction(self, pred):
        """Преобразует индексы нейросети в текст номера"""
        result = ""
        for char_probs in pred:
            char_idx = np.argmax(char_probs)
            if char_idx in self.num_to_char:
                result += self.num_to_char[char_idx]
        return result

    def run(self):
        """Основная логика распознавания"""
        try:
            img = cv2.imread(self.image_path)
            if img is None:
                raise Exception("Не удалось загрузить файл изображения")

            results = self.detector.predict(img, conf=CONFIDENCE_THRESHOLD, imgsz=640, device=0, verbose=False)
            found_plates = []

            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # Вырезание номера
                    h_i, w_i, _ = img.shape
                    crop = img[max(0, y1 - 5):min(h_i, y2 + 5), max(0, x1 - 5):min(w_i, x2 + 5)]

                    if crop.size == 0: 
                        continue

                    # Подготовка изображения
                    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                    resized = cv2.resize(gray, (IMG_WIDTH, IMG_HEIGHT))
                    normalized = resized.astype('float32') / 255.0
                    input_tensor = np.expand_dims(np.expand_dims(normalized, axis=-1), axis=0)

                    # Распознавание
                    prediction = self.reader.predict(input_tensor, verbose=0)[0]
                    plate_text = self.decode_prediction(prediction)
                    found_plates.append(plate_text)

                    # Рисование результата
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    cv2.rectangle(img, (x1, y1 - 35), (x1 + 180, y1), (0, 255, 0), -1)
                    cv2.putText(img, plate_text, (x1 + 5, y1 - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)

            # Конвертирование в RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w, ch = img_rgb.shape
            qt_image = QImage(img_rgb.data, w, h, ch * w, QImage.Format_RGB888)

            result_str = ", ".join(found_plates) if found_plates else "Номер не обнаружен"

            self.finished.emit({
                "image": qt_image,
                "text": result_str,
                "count": len(found_plates)
            })
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error.emit(str(e))


# ==================== ГЛАВНОЕ ОКНО ПРИЛОЖЕНИЯ ====================
class BlackwellLPRApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.detector = None
        self.reader = None
        self.initUI()
        self.load_models()

    def initUI(self):
        """Создание интерфейса"""
        self.setWindowTitle("Blackwell LPR (V2.1 - STABLE)")
        self.setMinimumSize(850, 750)

        self.setStyleSheet("""
            QMainWindow { background-color: #121212; }
            QLabel { color: #e0e0e0; font-family: 'Segoe UI'; }
            QPushButton {
                background-color: #222; color: #00ff00; border: 1px solid #333;
                border-radius: 10px; padding: 12px; font-weight: bold;
            }
            QPushButton:hover { background-color: #333; border-color: #00ff00; }
        """)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        title = QLabel(" Blackwell LPR System (Stable) ")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 28px; color: #00ff00; margin: 15px;")
        layout.addWidget(title)

        self.image_display = QPushButton("Нажмите, чтобы выбрать фото для анализа")
        self.image_display.setMinimumHeight(480)
        self.image_display.setStyleSheet("border: 2px dashed #444; color: #666; font-size: 18px;")
        self.image_display.clicked.connect(self.select_image)
        layout.addWidget(self.image_display)

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

        self.result_label = QLabel("Ожидание файла...")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet("font-size: 26px; font-weight: bold; color: #00ff00; padding: 20px;")
        layout.addWidget(self.result_label)

        self.statusBar().showMessage("Готов к работе")

    def load_models(self):
        """Загрузка нейросетей при запуске"""
        self.statusBar().showMessage("Загрузка систем распознавания... подождите")
        try:
            self.detector = YOLO(DETECTOR_PATH)
            self.reader = tf.keras.models.load_model(READER_PATH)
            self.statusBar().showMessage("Системы загружены")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить модели!\n{e}")

    def select_image(self):
        """Выбор файла на компьютере"""
        path, _ = QFileDialog.getOpenFileName(self, "Выбрать авто", "", "Изображения (*.jpg *.jpeg *.png)")
        if path:
            self.process_image(path)

    def process_image(self, path):
        """Запуск фонового потока анализа"""
        self.image_display.setEnabled(False)
        self.progress.setVisible(True)
        self.result_label.setText("Анализирую изображение...")

        self.thread = LPRThread(self.detector, self.reader, path)
        self.thread.finished.connect(self.on_success)
        self.thread.error.connect(self.on_fail)
        self.thread.start()

    def on_success(self, data):
        """Действия при успешном распознавании"""
        self.progress.setVisible(False)
        self.image_display.setEnabled(True)

        pixmap = QPixmap.fromImage(data["image"])
        scaled = pixmap.scaled(self.image_display.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_display.setIcon(QIcon(scaled))
        self.image_display.setIconSize(scaled.size())
        self.image_display.setText("")

        self.result_label.setText(f"Результат: {data['text']}")
        self.statusBar().showMessage(f"Обработка завершена. Найдено: {data['count']}")

    def on_fail(self, msg):
        """Если произошла ошибка"""
        self.progress.setVisible(False)
        self.image_display.setEnabled(True)
        QMessageBox.warning(self, "Ошибка анализа", msg)


# ==================== ЗАПУСК ====================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = BlackwellLPRApp()
    window.show()
    sys.exit(app.exec_())
