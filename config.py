# config.py - Очищенная конфигурация
import os
import tkinter as tk

# Определяем размер экрана
def get_screen_info():
    """Получение информации о экране"""
    try:
        root = tk.Tk()
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        root.destroy()
        return screen_width, screen_height
    except:
        return 1920, 1080  # По умолчанию

SCREEN_WIDTH, SCREEN_HEIGHT = get_screen_info()

# Основные настройки
APP_NAME = "Двухфакторная аутентификация"
VERSION = "1.2.0"

# Пути к файлам
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(DATA_DIR, "models")
DATABASE_PATH = os.path.join(DATA_DIR, "users.db")

# Создание директорий
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Настройки машинного обучения
MIN_TRAINING_SAMPLES = 50
DEFAULT_KNN_NEIGHBORS = 5  # Заменили KNN_NEIGHBORS
DEFAULT_THRESHOLD = 0.75

# Адаптивные настройки GUI
if SCREEN_WIDTH >= 1920 and SCREEN_HEIGHT >= 1080:
    WINDOW_WIDTH = 800
    WINDOW_HEIGHT = 900
    TRAINING_WINDOW_WIDTH = 900
    TRAINING_WINDOW_HEIGHT = 1000
    STATS_WINDOW_WIDTH = 1400
    STATS_WINDOW_HEIGHT = 900
    FONT_SIZE = 11
elif SCREEN_WIDTH >= 1366:
    WINDOW_WIDTH = 700
    WINDOW_HEIGHT = 800
    TRAINING_WINDOW_WIDTH = 800
    TRAINING_WINDOW_HEIGHT = 900
    STATS_WINDOW_WIDTH = 1200
    STATS_WINDOW_HEIGHT = 800
    FONT_SIZE = 10
else:
    WINDOW_WIDTH = 600
    WINDOW_HEIGHT = 700
    TRAINING_WINDOW_WIDTH = 700
    TRAINING_WINDOW_HEIGHT = 800
    STATS_WINDOW_WIDTH = 1000
    STATS_WINDOW_HEIGHT = 700
    FONT_SIZE = 9

FONT_FAMILY = "Arial"

# Настройки безопасности
SALT_LENGTH = 32

# Панграмма для обучения и аутентификации
PANGRAM = "The quick brown fox jumps over the lazy dog"

# Пути для дополнительных данных
TEMP_DIR = os.path.join(DATA_DIR, "temp")
CSV_EXPORTS_DIR = os.path.join(DATA_DIR, "csv_exports")

# Создание дополнительных директорий
os.makedirs(TEMP_DIR, exist_ok=True) 
os.makedirs(CSV_EXPORTS_DIR, exist_ok=True)

# Параметры модели по умолчанию
DEFAULT_KNN_PARAMS = {
    'n_neighbors': 5,
    'weights': 'uniform',
    'metric': 'euclidean',
    'algorithm': 'auto'
}

# Отладочная информация
DEBUG_MODE = True
if DEBUG_MODE:
    print(f"📋 {APP_NAME} v{VERSION}")
    print(f"🖥️  Экран: {SCREEN_WIDTH}x{SCREEN_HEIGHT}")
    print(f"📐 Окно: {WINDOW_WIDTH}x{WINDOW_HEIGHT}")
    print(f"📁 Данные: {DATA_DIR}")