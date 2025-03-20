import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from CNN_Realization import create_cnn_model, load_and_prepare_data

def test_model_creation():
    print("\nТест создания модели CNN...")
    model = create_cnn_model(num_classes=2)
    print(f"✓ Модель создана успешно")
    print(f"✓ Количество слоев: {len(model.layers)}")
    return True

def test_data_loading():
    print("\nТест загрузки данных...")
    (X_train, y_train), (X_test, y_test), classes = load_and_prepare_data()
    
    print(f"✓ Размер обучающей выборки: {X_train.shape}")
    print(f"✓ Размер тестовой выборки: {X_test.shape}")
    print(f"✓ Количество классов: {len(classes)}")
    print(f"✓ Классы: {classes}")
    return True

def test_model_prediction():
    """Тест предсказаний модели"""
    print("\nТест предсказаний модели...")
    # Загрузка данных
    (X_train, y_train), (X_test, y_test), classes = load_and_prepare_data()
    # Создание модели
    model = create_cnn_model(num_classes=len(classes))
    
    # Тестовое предсказание
    test_image = X_test[0:1]
    prediction = model.predict(test_image)
    print(f"✓ Форма предсказания: {prediction.shape}")
    print(f"✓ Диапазон значений: [{np.min(prediction):.3f}, {np.max(prediction):.3f}]")
    return True

def test_model_on_images():
    print("\nТест модели на конкретных изображениях...")
    # Загрузка обученной модели
    model_path = 'anime_cartoon_cnn_model.h5'
    model = tf.keras.models.load_model(model_path)
    
    # Загрузка данных для получения классов
    (X_train, y_train), (X_test, y_test), classes = load_and_prepare_data()
    
    # Путь к датасету
    dataset_path = os.path.join(os.path.dirname(__file__), 'data', 'Training Data')
    
    # Создание подграфиков с большим размером и высоким DPI
    plt.figure(figsize=(20, 20), dpi=300)
    
    # Создание подграфиков
    fig, axes = plt.subplots(2, 2, figsize=(20, 20))
    axes = axes.ravel()
    
    # Счетчик для подграфиков
    plot_idx = 0
    
    # Проход по каждому классу
    for class_idx, class_name in enumerate(classes):
        class_path = os.path.join(dataset_path, class_name)
        
        if not os.path.exists(class_path):
            continue
            
        # Собираем все изображения из всех подпапок
        all_images = []
        for root, _, files in os.walk(class_path):
            for file in files:
                if file.endswith('.png'):
                    all_images.append(os.path.join(root, file))
        
        if not all_images:
            continue
            
        # Выбор двух случайных изображений для каждого класса
        selected_images = np.random.choice(all_images, size=2, replace=False)
        
        for img_path in selected_images:
            print(f"✓ Обработка изображения: {img_path}")
            
            # Загрузка и предобработка изображения
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_original = img.copy()
            img = cv2.resize(img, (64, 64))
            img = img.astype('float32') / 255.0
            img = np.expand_dims(img, axis=0)
            
            # Получение предсказания
            prediction = model.predict(img)
            predicted_class = classes[np.argmax(prediction)]
            confidence = np.max(prediction) * 100
            
            # Отображение изображения и результатов
            axes[plot_idx].imshow(img_original)
            axes[plot_idx].set_title(f'Истинный класс: {class_name}\nПредсказанный класс: {predicted_class}\nУверенность: {confidence:.2f}%', 
                                   fontsize=14, pad=20)
            axes[plot_idx].axis('off')
            
            plot_idx += 1
    
    plt.tight_layout()
    # Сохранение графика в файл с высоким качеством
    plt.savefig('model_test_results.png', 
                dpi=300, 
                bbox_inches='tight', 
                pad_inches=0.5)
    plt.close()
    print("✓ Результаты сохранены в файл 'model_test_results.png'")
    return True

def test_model_on_no_dataset():
    """Тест модели на изображениях из папки No dataset Data"""
    print("\nТест модели на изображениях из No dataset Data...")
    
    # Загрузка обученной модели
    model_path = os.path.join(os.path.dirname(__file__), 'anime_cartoon_cnn_model.h5')
    model = tf.keras.models.load_model(model_path)
    
    # Загрузка данных для получения классов
    (X_train, y_train), (X_test, y_test), classes = load_and_prepare_data()
    
    # Путь к папке с тестовыми изображениями
    test_path = os.path.join(os.path.dirname(__file__), 'data', 'No dataset Data', 'Cartoon')
    
    # Получение списка изображений
    test_images = []
    for img_name in os.listdir(test_path):
        if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            test_images.append(os.path.join(test_path, img_name))
    
    if not test_images:
        print("✗ Не найдены изображения для тестирования")
        return False
    
    print(f"✓ Найдено изображений: {len(test_images)}")
    
    # Создание подграфиков
    n_images = len(test_images)
    n_cols = min(3, n_images)
    n_rows = (n_images + n_cols - 1) // n_cols
    
    plt.figure(figsize=(20, 8 * n_rows), dpi=300)
    
    # Обработка каждого изображения
    for idx, img_path in enumerate(test_images):
        print(f"✓ Обработка изображения: {img_path}")
        
        # Загрузка и предобработка изображения
        img = cv2.imread(img_path)
        if img is None:
            print(f"✗ Не удалось загрузить изображение: {img_path}")
            continue
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_original = img.copy()
        img = cv2.resize(img, (64, 64))
        img = img.astype('float32') / 255.0
        img = np.expand_dims(img, axis=0)
        
        # Получение предсказания
        prediction = model.predict(img, verbose=0)  # Отключаем вывод прогресса
        predicted_class = classes[np.argmax(prediction)]
        confidence = np.max(prediction) * 100
        
        # Отображение изображения и результатов
        plt.subplot(n_rows, n_cols, idx + 1)
        plt.imshow(img_original)
        plt.title(f'Изображение: {os.path.basename(img_path)}\nПредсказанный класс: {predicted_class}\nУверенность: {confidence:.2f}%', 
                 fontsize=14, pad=20)
        plt.axis('off')
    
    plt.tight_layout()
    
    # Сохранение результатов
    plt.savefig('no_dataset_test_results.png', 
                dpi=300, 
                bbox_inches='tight', 
                pad_inches=0.5)
    plt.close()
    print("✓ Результаты сохранены в файл 'no_dataset_test_results.png'")
    return True

def main():
    print("Начало тестирования модели на No dataset Data...")
    test_model_on_no_dataset()
    print("\nТестирование завершено!")

if __name__ == "__main__":
    main()
