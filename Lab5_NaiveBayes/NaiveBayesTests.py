from matplotlib import pyplot as plt
from scipy.stats import norm, laplace
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from NaiveBayesRealization import *
from ucimlrepo import fetch_ucirepo 
  
# Получение датасета
#dry_bean = fetch_ucirepo(id = 602)
dry_bean = fetch_ucirepo(id = 53) 
  
# Данные (pandas dataframes) 
X = dry_bean.data.features.values
y = dry_bean.data.targets.values
features = dry_bean.data.features.head()
print(features)

y = y.ravel() # Преобразует (n, 1) в (n,) двумерный в одномерный

# Стандартизация
scaler = StandardScaler()
X = scaler.fit_transform(X)
  
# Метадата и информация о переменных
print(dry_bean.metadata) 
print(dry_bean.variables) 

# Разделение выборки на тренировочную и тестовую
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

# Из библиотеки sklearn
g = GaussianNB()
g.fit(X_train, y_train)
predictions = g.predict(X_test)
accuracy = np.mean(predictions == y_test)
print(f"Точность Guassian модели из sklearn для test: {accuracy * 100:.2f}%")

g = GuassianNaiveBayes(X_train, y_train)

predictions = g.predict(X_train)

# Оценка точности
accuracy = np.mean(predictions == y_train)
print(f"Точность Guassian модели для train: {accuracy * 100:.2f}%")

predictions = g.predict(X_test)

# Оценка точности
accuracy = np.mean(predictions == y_test)
print(f"Точность Guassian модели для test: {accuracy * 100:.2f}%")

l = LaplacianNaiveBayes(X_train, y_train)

predictions = l.predict(X_train)

# Оценка точности
accuracy = np.mean(predictions == y_train)
print(f"Точность Laplacian модели для train: {accuracy * 100:.2f}%")

predictions = l.predict(X_test)

# Оценка точности
accuracy = np.mean(predictions == y_test)
print(f"Точность Laplacian модели для test: {accuracy * 100:.2f}%")


def plot_distributions(X, y, feature_idx, class_idx, nb_normal, nb_laplace, class_label):
    X_c = X[y == class_idx, feature_idx]

    if len(X_c) == 0:
        print(f"Для класса {class_label} и признака {feature_idx} нет данных.")
        return  # Выход из функции, если данных нет
    
    # Эмпирическое распределение
    plt.hist(X_c, bins=15, density=True, alpha=0.5, label='Эмпирическое')
    
    # Теоретическое нормальное распределение
    M, D = nb_normal.M[class_idx, feature_idx], nb_normal.D[class_idx, feature_idx]
    x = np.linspace(min(X_c), max(X_c), 100)
    normal_pdf = (1/np.sqrt(2 * np.pi * D)) * np.exp(-(x - M)**2 / (2 * D))
    plt.plot(x, normal_pdf, label='Нормальное распределение', color='blue')
    
    # Теоретическое лапласовское распределение
    median, scale = nb_laplace.median[class_idx, feature_idx], nb_laplace.scale[class_idx, feature_idx]
    laplace_pdf = (1/(2 * scale)) * np.exp(-np.abs(x - median) / scale)
    plt.plot(x, laplace_pdf, label='Лапласовское распределение', color='red')
    
    plt.title(f'Признак {feature_idx} - Класс {class_label}')
    plt.xlabel(features[feature_idx])
    plt.ylabel('Плотность')
    plt.legend()
    plt.show()

# Построение графиков для всех признаков для одного класса (например, класса 0)
for feature_idx in range(X_train.shape[1]):  # Для всех признаков
    plot_distributions(X_train, y_train, feature_idx, 0, g, l, '0')