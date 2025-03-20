import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from CNN_Realization import create_cnn_model, get_classes, IMG_SIZE


def main():
    print("Начало тестирования...")

    # Загрузка обученной модели
    model_path = 'anime_cartoon_cnn_model.h5'
    model = tf.keras.models.load_model(model_path)
    
    # Путь к датасету
    dataset_path = os.path.join(os.path.dirname(__file__), 'data', 'Training Data')
    
    # Классы
    classes = get_classes(dataset_path)

    # Создание подграфиков
    plt.figure(figsize=(20, 20), dpi=300)
    fig, axes = plt.subplots(2, 2, figsize=(20, 20))
    axes = axes.ravel()

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
                if file.endswith(('.png', '.jpg')):
                    all_images.append(os.path.join(root, file))
        
        if not all_images:
            continue
            
        # Выбор двух случайных изображений для каждого класса
        selected_images = np.random.choice(all_images, size=2, replace=False)
        
        for img_path in selected_images:
            print(f"Обработка изображения: {img_path}")
            
            # Загрузка и предобработка изображения
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_original = img.copy()
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
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
    print("Результаты сохранены в файл 'model_test_results.png'")


if __name__ == "__main__":
    main()
