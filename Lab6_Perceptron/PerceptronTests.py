from multiprocessing import Process

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt

import PerceptronRealization as Realization

from ucimlrepo import fetch_ucirepo

def main():
    students = fetch_ucirepo(id=320)  # Загружаем dataset

    # Выводим информацию о доступных признаках
    print(students.data.features.head())
    print(students.metadata)

    # Выбираем признаки для регрессии
    selected_features = ['famrel', 'age', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences']
    X = students.data.features[selected_features].values

    # Проверяем целевую переменную
    y = students.data.targets['G3'].values
    y_scaler = StandardScaler()
    y = y_scaler.fit_transform(y.reshape(-1, 1)).ravel()

    # Стандартизация признаков
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X = X.astype(np.float32)
    y = y.astype(np.float32)

    # Разделение выборки на тренировочную и тестовую
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Объекты нейросетей
    XORSigmoid  = Realization.Neural('sigmoid', len(selected_features), 0.001, 800)
    XORLinear   = Realization.Neural('linear', len(selected_features), 0.001, 800)
    XORReLu     = Realization.Neural('relu', len(selected_features), 0.001, 800)

    # Обучение в разных потоках
    t1 = Process(target=XORSigmoid.train, args=(X_train, y_train))
    t2 = Process(target=XORLinear.train, args=(X_train, y_train))
    t3 = Process(target=XORReLu.train, args=(X_train, y_train))

    # Запуск потоков
    t1.start()
    t2.start()
    t3.start()

    # Ожидание обучения
    t1.join()
    t2.join()
    t3.join()

    # Сохранение ответов
    predictSigmoid  = XORSigmoid.predict(X_test)
    predictLinear   = XORLinear.predict(X_test)
    predictReLu     = XORReLu.predict(X_test)

    # Денормализация
    predictSigmoid = y_scaler.inverse_transform(predictSigmoid)
    predictLinear = y_scaler.inverse_transform(predictLinear)
    predictReLu = y_scaler.inverse_transform(predictReLu)

    # Денормализация y_test для дальнейшего сравнения
    y_test_denorm = y_scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()

    mseSigmoid  = mean_squared_error(y_test, predictSigmoid)
    mseLinear   = mean_squared_error(y_test, predictLinear)
    mseReLu     = mean_squared_error(y_test, predictReLu)

    # Вывод результатов
    print(f"Sigmoid mse {mseSigmoid}")

    print(f"Linear mse {mseLinear}")

    print(f"ReLu mse {mseReLu}")

    plt.figure(figsize=(10, 6))
    plt.plot(y_test_denorm, label="Действительное значение", linestyle='--')
    plt.plot(predictSigmoid, label="Sigmoid Prediction")
    plt.plot(predictLinear, label="Linear Prediction")
    plt.plot(predictReLu, label="ReLU Prediction")
    plt.legend()
    plt.title("Предсказанное vs Действительное")
    plt.xlabel("Объект")
    plt.ylabel("Итоговая оценка")
    plt.show()


if __name__ == "__main__":
    main()


