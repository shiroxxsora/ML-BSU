import threading
import XORTensorFlowRealizationKeras
import numpy as np

def main():

    # Данные
    X = np.array([[0,0], [0,1], [1,0], [1,1]], dtype=np.float32)
    Y = np.array([[0], [1], [1], [0]], dtype=np.float32)

    # Объекты нейросетей
    XORSigmoid  = XORTensorFlowRealizationKeras.XORSigmoid()
    XORLinear   = XORTensorFlowRealizationKeras.XORLinear()
    XORReLu     = XORTensorFlowRealizationKeras.XORReLu()

    # Обучение в разных потоках
    t1 = threading.Thread(target = XORSigmoid.train, args=(X, Y))
    t2 = threading.Thread(target = XORLinear.train, args=(X, Y))
    t3 = threading.Thread(target = XORReLu.train, args=(X, Y))

    # Запуск потоков
    t1.start()
    t2.start()
    t3.start()

    # #Сохранение ответов
    # predictSigmoid  = XORSigmoid.predict(X)
    # predictLinear   = XORLinear.predict(X)
    # predictReLu     = XORReLu.predict(X)

    # #Вывод результатов
    # for i in range(len(X)):
    #     print(f"X = {X[i]}\n\tSigmoid y = {predictSigmoid[i]}\n\tLinear y = {predictLinear[i]}\n\tReLu y = {predictReLu[i]}")


if __name__ == "__main__":
    main()
