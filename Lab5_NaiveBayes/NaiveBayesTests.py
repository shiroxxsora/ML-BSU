from matplotlib import pyplot as plt
from scipy.stats import norm, laplace
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from NaiveBayesRealization import *
from ucimlrepo import fetch_ucirepo 
  
# Получение датасета
dry_bean = fetch_ucirepo(id = 602)
#dry_bean = fetch_ucirepo(id = 53) 
  
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

# Выбираем 1 класс 
class_index = 0
X_class = X[y == np.unique(y)[class_index]]

# Выбираем один признак для анализа
feature_index = 0
feature_data = X_class[:, feature_index]

# Эмпирическое распределение (гистограмма)
plt.hist(feature_data, bins=30, density=True, alpha=0.6, color='g', label='Эмпирическое распределение')

# Теоретическое нормальное распределение
mean = np.mean(feature_data)
std = np.std(feature_data)
x = np.linspace(np.min(feature_data), np.max(feature_data), 1000)
normal_pdf = norm.pdf(x, mean, std)
plt.plot(x, normal_pdf, 'r-', label=f'Нормальное распределение (μ={mean:.2f}, σ={std:.2f})')

# Теоретическое Лаплассовское распределение
median = np.median(feature_data)
scale = np.median(np.abs(feature_data - median))  # Масштаб Лапласа (MAD)
laplace_pdf = laplace.pdf(x, median, scale)
plt.plot(x, laplace_pdf, 'b-', label=f'Лаплассово распределение (медиана={median:.2f}, масштаб={scale:.2f})')

# Добавляем легенду
plt.legend()

# Добавляем подписи
plt.title(f'Сравнение эмпирического и теоретических распределений для признака {feature_index}')
plt.xlabel(f'Признак {feature_index}')
plt.ylabel('Плотность вероятности')

# Показываем график
plt.show()
