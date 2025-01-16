import threading
import XORTensorFlowRealization as XORRealization
import numpy as np

def main():

    # Данные
    X = np.array([[0,0], [0,1], [1,0], [1,1]], dtype=np.float32)
    Y = np.array([[0], [1], [1], [0]], dtype=np.float32)

    # Объекты нейросетей
    XORSigmoid  = XORRealization.XORNeural('sigmoid', 0.1, 400)
    XORLinear   = XORRealization.XORNeural('linear', 0.1, 400)
    XORReLu     = XORRealization.XORNeural('relu', 0.1, 400)

    # Обучение в разных потоках
    t1 = threading.Thread(target = XORSigmoid.train, args=(X, Y))
    t2 = threading.Thread(target = XORLinear.train, args=(X, Y))
    t3 = threading.Thread(target = XORReLu.train, args=(X, Y))

    # Запуск потоков
    t1.start()
    t2.start()
    t3.start()

    # Ожидание обучения
    t1.join()
    t2.join()
    t3.join()

    # Сохранение ответов
    predictSigmoid  = XORSigmoid.predict(X)
    predictLinear   = XORLinear.predict(X)
    predictReLu     = XORReLu.predict(X)

    # Вывод результатов
    print("Sigmoid")
    for i in range(len(X)):
        print(f"X = {X[i]} Predicted = {predictSigmoid[i]} Actual = {Y[i]}")

    print("Linear")
    for i in range(len(X)):
        print(f"X = {X[i]} Predicted = {predictLinear[i]} Actual = {Y[i]}")

    print("ReLu")
    for i in range(len(X)):
        print(f"X = {X[i]} Predicted = {predictReLu[i]} Actual = {Y[i]}")


if __name__ == "__main__":
    main()
