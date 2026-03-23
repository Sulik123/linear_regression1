import pandas as pd
import numpy as np
import joblib

model = joblib.load('medical_model.pkl')
features = joblib.load('features.pkl')

def predict(age, bmi, children, smoker, sex=None, region=None):
    smoker_val = 1 if smoker == 'yes' else 0
    
    input_data = pd.DataFrame([[age, bmi, children, smoker_val,
                                bmi * smoker_val,
                                age * bmi,
                                age ** 2,
                                bmi ** 2,
                                children ** 2,
                                age * smoker_val]],
                              columns=features)
    
    pred_log = model.predict(input_data)[0]
    return np.expm1(pred_log)

while True:
    print("\n--- Введите данные ---")
    
    while True:
        age_input = input("Возраст: ").strip()
        if age_input.lower() == 'exit':
            print("Выход из программы")
            exit()
        try:
            age = int(age_input)
            if 18 <= age <= 100:
                break
            else:
                print("Возраст должен быть от 18 до 100 лет")
        except ValueError:
            print("Пожалуйста, введите число. Пример: 35")
    
    while True:
        bmi_input = input("ИМТ (BMI): ").strip()
        if bmi_input.lower() == 'exit':
            print("Выход из программы")
            exit()
        try:
            bmi = float(bmi_input)
            if 10 <= bmi <= 60:
                break
            else:
                print("ИМТ должен быть от 10 до 60")
        except ValueError:
            print("введите число. Пример: 28.5")
    
    while True:
        children_input = input("Детей: ").strip()
        if children_input.lower() == 'exit':
            print("Выход из программы")
            exit()
        try:
            children = int(children_input)
            if 0 <= children <= 10:
                break
            else:
                print("Количество детей должно быть от 0 до 10")
        except ValueError:
            print("Пожалуйста, введите целое число. Пример: 2")
    
    while True:
        smoker_input = input("Курит (yes/no): ").strip().lower()
        if smoker_input == 'exit':
            print("Выход из программы")
            exit()
        if smoker_input in ['yes', 'no']:
            smoker = smoker_input
            break
        else:
            print("Пожалуйста, введите 'yes' или 'no'")
    
    try:
        result = predict(age, bmi, children, smoker)
        print(f"\n Расходы: ${result:,.2f}")
    except Exception as e:
        print(f"Ошибка при предсказании: {e}")
    while True:
        again = input("\nЕщё? (y/n): ").strip().lower()
        if again == 'y':
            break
        elif again == 'n' or again == 'exit':
            print("выход из программы")
            exit()
        else:
            print("Введите 'y' (да) или 'n' (нет)")