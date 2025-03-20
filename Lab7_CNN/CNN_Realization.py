import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
import cv2
import kagglehub

IMG_SIZE = 192  # Размер изображения для сети

# Проверка доступных устройств
print("Доступные устройства:")
print(tf.config.list_physical_devices())

# Настройка GPU
try:
    # Получаем список доступных GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        # Включаем память GPU по требованию
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU доступен и настроен")
    else:
        print("GPU не найден, используется CPU")
except Exception as e:
    print(f"Ошибка при настройке GPU: {e}")

def get_classes(dataset_path):
    return os.listdir(dataset_path)

# Загрузка и подготовка данных
def load_and_prepare_data():
    # Путь к локальному датасету
    dataset_path = os.path.join(os.path.dirname(__file__), 'data', 'Training Data')
    print("Путь к датасету:", dataset_path)
    
    # Получение списка классов (папок)
    classes = get_classes(dataset_path)
    print("Классы:", classes)
    
    images = []
    labels = []
    
    # Загрузка изображений и их меток
    for class_idx, class_name in enumerate(classes):
        class_path = os.path.join(dataset_path, class_name)
        print(f"Загрузка класса {class_name}...")
        
        # Получение списка подпапок в классе
        subfolders = os.listdir(class_path)
        for subfolder in subfolders:
            subfolder_path = os.path.join(class_path, subfolder)
            if os.path.isdir(subfolder_path):
                # Получение списка изображений в подпапке
                for img_name in os.listdir(subfolder_path):
                    if img_name.endswith(('.png', '.jpg')):
                        img_path = os.path.join(subfolder_path, img_name)
                        try:
                            # Чтение и предобработка изображения
                            img = cv2.imread(img_path)
                            if img is None:
                                print(f"Не удалось загрузить изображение: {img_path}")
                                continue
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                            img = img.astype('float32') / 255.0
                            
                            images.append(img)
                            labels.append(class_idx)
                        except Exception as e:
                            print(f"Ошибка при загрузке {img_path}: {str(e)}")
    
    print(f"Загружено изображений: {len(images)}")

    X = np.array(images)
    y = np.array(labels)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # One-hot encoding для меток
    y_train = tf.keras.utils.to_categorical(y_train, len(classes))
    y_test = tf.keras.utils.to_categorical(y_test, len(classes))
    
    return (X_train, y_train), (X_test, y_test), classes


def create_cnn_model(num_classes, img_size):
    model = models.Sequential([
        # Первый сверточный блок
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_size, img_size, 3),
                     kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),
        
        # Второй сверточный блок
        layers.Conv2D(64, (3, 3), activation='relu',
                     kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),
        
        # Третий сверточный блок
        layers.Conv2D(64, (3, 3), activation='relu',
                     kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.4),
        
        # Полносвязные слои
        layers.Flatten(),
        layers.Dense(128, activation='relu',
                    kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

# Создание генератора аугментации данных
def create_data_generator():
    return tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2
    )


def train_model(model, x_train, y_train, x_test, y_test):
    # Компиляция модели
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])
    
    datagen = create_data_generator()
    
    # Настройка ранней остановки
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
    
    # Настройка уменьшения скорости обучения
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        min_lr=0.00001
    )
    
    # Обучение модели с аугментацией данных
    history = model.fit(
        datagen.flow(x_train, y_train, batch_size=32, subset='training'),
        validation_data=datagen.flow(x_train, y_train, batch_size=32, subset='validation'),
        epochs=30,
        callbacks=[early_stopping, reduce_lr]
    )
    
    return history

# Функция для визуализации результатов обучения
def plot_training_results(history):
    plt.figure(figsize=(20, 8), dpi=300)
    
    # График точности
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Точность на обучающей выборке', linewidth=2)
    plt.plot(history.history['val_accuracy'], label='Точность на тестовой выборке', linewidth=2)
    plt.title('Точность модели', fontsize=14, pad=20)
    plt.xlabel('Эпоха', fontsize=12)
    plt.ylabel('Точность', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # График потерь
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Потери на обучающей выборке', linewidth=2)
    plt.plot(history.history['val_loss'], label='Потери на тестовой выборке', linewidth=2)
    plt.title('Потери модели', fontsize=14, pad=20)
    plt.xlabel('Эпоха', fontsize=12)
    plt.ylabel('Потери', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    # Сохранение графиков в файл с высоким качеством
    plt.savefig('training_history.png', 
                dpi=300, 
                bbox_inches='tight', 
                pad_inches=0.5)
    plt.close()
    print("Графики обучения сохранены в файл 'training_history.png'")

def main():
    # Загрузка данных
    print("Загрузка данных...")
    (x_train, y_train), (x_test, y_test), classes = load_and_prepare_data()
    
    print(f"Размер обучающей выборки: {x_train.shape}")
    print(f"Размер тестовой выборки: {x_test.shape}")
    print(f"Количество классов: {len(classes)}")
    
    # Создание модели
    print("Создание модели CNN...")
    model = create_cnn_model(len(classes), IMG_SIZE)
    model.summary()
    
    # Обучение модели
    print("Начало обучения модели...")
    history = train_model(model, x_train, y_train, x_test, y_test)
    
    # Визуализация результатов
    print("Визуализация результатов обучения...")
    plot_training_results(history)
    
    # Сохранение модели
    print("Сохранение модели...")
    model.save('anime_cartoon_cnn_model.h5')
    
    # Оценка модели на тестовых данных
    test_loss, test_accuracy = model.evaluate(x_test, y_test)
    print(f"\nТочность на тестовых данных: {test_accuracy:.4f}")
    print(f"Потери на тестовых данных: {test_loss:.4f}")

if __name__ == "__main__":
    main()
