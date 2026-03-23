import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

plt.ioff()

data = pd.read_csv('insurance.csv')
data['smoker'] = data['smoker'].map({'yes': 1, 'no': 0})
data['sex'] = data['sex'].map({'male': 1, 'female': 0})
data['region'] = data['region'].astype('category').cat.codes

data['bmi_smoker'] = data['bmi'] * data['smoker']
data['age_bmi'] = data['age'] * data['bmi']
data['age2'] = data['age'] ** 2
data['bmi2'] = data['bmi'] ** 2
data['children2'] = data['children'] ** 2
data['age_smoker'] = data['age'] * data['smoker']

y = np.log1p(data['charges'])

features = ['age', 'bmi', 'children', 'smoker', 
            'bmi_smoker', 'age_bmi', 'age2', 'bmi2', 
            'children2', 'age_smoker']
X = data[features]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = model.score(X_test, y_test) * 100
print(f"Точность на тесте: {accuracy:.1f}%")

joblib.dump(model, 'medical_model.pkl')
joblib.dump(features, 'features.pkl')

y_test_orig = np.expm1(y_test)
y_pred_orig = np.expm1(y_pred)

plt.figure(figsize=(10, 6))

plt.scatter(y_test_orig, y_pred_orig, alpha=0.5, label='Предсказания')

min_val = min(y_test_orig.min(), y_pred_orig.min())
max_val = max(y_test_orig.max(), y_pred_orig.max())
plt.plot([y_test_orig.min(), y_test_orig.max()], [y_pred_orig.min(), y_pred_orig.max()], 
         'g--', linewidth=2, label='Идеал (y = x)')

plt.xlabel('Реальные цены')
plt.ylabel('Предсказанные цены')
plt.title(f'Точность модели: {accuracy:.1f}%')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show(block=True)