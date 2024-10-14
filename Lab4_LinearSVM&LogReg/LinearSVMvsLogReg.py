# Импортируем необходимые библиотеки
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer

# Загружаем датасет (двухклассовая задача - рак молочной железы)
data = load_breast_cancer()

# Преобразуем в DataFrame для удобства
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

# Разделим выборку на признаки и целевую переменную
X = df.drop(columns='target')
y = df['target']

# Стандартизация признаков
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Разделим данные на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Инициализируем модели
log_reg = LogisticRegression(solver='liblinear', max_iter=10000)
svc = LinearSVC(max_iter=10000)

# Обучение моделей
log_reg.fit(X_train, y_train)
svc.fit(X_train, y_train)

# Предсказания и оценка accuracy
log_reg_pred = log_reg.predict(X_test)
svc_pred = svc.predict(X_test)

log_reg_acc = accuracy_score(y_test, log_reg_pred)
svc_acc = accuracy_score(y_test, svc_pred)

print(f"Accuracy Logistic Regression: {log_reg_acc}")
print(f"Accuracy Linear SVC: {svc_acc}")

# Настроим диапазон значений C
C_values = [0.01, 0.1, 1, 10, 100]
log_reg_accuracies = []
svc_accuracies = []

# Пробуем разные значения C для обеих моделей
for C in C_values:
    # Логистическая регрессия
    log_reg = LogisticRegression(C=C, solver='liblinear', max_iter=10000)
    log_reg.fit(X_train, y_train)
    log_reg_pred = log_reg.predict(X_test)
    log_reg_accuracies.append(accuracy_score(y_test, log_reg_pred))
    
    # SVM
    svc = LinearSVC(C=C, max_iter=10000)
    svc.fit(X_train, y_train)
    svc_pred = svc.predict(X_test)
    svc_accuracies.append(accuracy_score(y_test, svc_pred))

# Построение графиков зависимости accuracy от C
plt.figure(figsize=(10, 6))
plt.plot(C_values, log_reg_accuracies, label="Logistic Regression", marker='o')
plt.plot(C_values, svc_accuracies, label="Linear SVC", marker='o')
plt.xscale('log')
plt.xlabel('C (log scale)')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Regularization Parameter C')
plt.legend()
plt.grid(True)
plt.show()

# Визуализация коэффициентов моделей при различных значениях C
coef_log_reg = []
coef_svc = []

for C in C_values:
    # Логистическая регрессия
    log_reg = LogisticRegression(C=C, solver='liblinear', max_iter=10000)
    log_reg.fit(X_train, y_train)
    coef_log_reg.append(log_reg.coef_[0])
    
    # SVM
    svc = LinearSVC(C=C, max_iter=10000)
    svc.fit(X_train, y_train)
    coef_svc.append(svc.coef_[0])

# Преобразуем в массивы для удобства построения графиков
coef_log_reg = np.array(coef_log_reg)
coef_svc = np.array(coef_svc)

# Построим графики коэффициентов
plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.plot(C_values, coef_log_reg[:, :5], marker='o')
plt.xscale('log')
plt.title('Logistic Regression Coefficients')
plt.xlabel('C (log scale)')
plt.ylabel('Coefficient Values')

plt.subplot(2, 1, 2)
plt.plot(C_values, coef_svc[:, :5], marker='o')
plt.xscale('log')
plt.title('Linear SVC Coefficients')
plt.xlabel('C (log scale)')
plt.ylabel('Coefficient Values')

plt.tight_layout()
plt.show()