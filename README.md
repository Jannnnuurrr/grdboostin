import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Загрузка фиктивных данных о университетах и исследовательских институтах в Казахстане
data = pd.DataFrame({
    'ResearchExpenditure': [100, 200, 300, 150, 250, 350, 120, 220, 180, 280],
    'FacultyQuality': [8, 7, 9, 6, 8, 9, 7, 8, 6, 7],
    'PublicationRate': [50, 60, 70, 40, 70, 80, 45, 55, 50, 60],
    'HighImpactJournals': [10, 12, 15, 8, 14, 17, 9, 11, 10, 13],
    'TopResearchers': [5, 6, 7, 4, 6, 8, 4, 5, 4, 6],
    'UniversityType': ['Public', 'Private', 'Public', 'Private', 'Public', 'Private', 'Public', 'Private', 'Public', 'Private']
})

# Кодируем категориальную переменную 'UniversityType' в числовой формат
data = pd.get_dummies(data, columns=['UniversityType'], drop_first=True)

# Определение целевой переменной
target = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  # Пример целевой переменной, где 1 - успешные учреждения, 0 - неуспешные

# Разделение данных на обучающий и тестовый набор
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)

# Инициализация и обучение градиентного бустинга
gb_classifier = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
gb_classifier.fit(X_train, y_train)

# Прогнозирование
predictions = gb_classifier.predict(X_test)

# Оценка модели
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy * 100:.2f}%")
