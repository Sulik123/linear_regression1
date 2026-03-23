from flask import Flask, render_template_string, request
import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import base64
from io import BytesIO

app = Flask(__name__)

model = joblib.load('medical_model.pkl')
features = joblib.load('features.pkl')

data = pd.read_csv('insurance.csv')
data_display = data.copy()

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Medical Cost Predictor</title>
    <meta charset="utf-8">
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 40px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }
        .stats {
            background-color: #ecf0f1;
            padding: 15px;
            border-radius: 8px;
            margin: 20px 0;
            display: flex;
            gap: 30px;
            flex-wrap: wrap;
        }
        .stat-card {
            background: white;
            padding: 10px 20px;
            border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        .stat-card h3 {
            margin: 0;
            color: #7f8c8d;
            font-size: 14px;
        }
        .stat-card p {
            margin: 5px 0 0;
            font-size: 24px;
            font-weight: bold;
            color: #2c3e50;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        th, td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }
        th {
            background-color: #3498db;
            color: white;
        }
        tr:nth-child(even) {
            background-color: #f9f9f9;
        }
        .form-group {
            margin: 15px 0;
        }
        label {
            display: inline-block;
            width: 120px;
            font-weight: bold;
        }
        input[type="number"] {
            padding: 8px;
            width: 200px;
            border: 1px solid #ddd;
            border-radius: 4px;
        }
        .radio-group {
            display: inline-block;
        }
        .radio-group label {
            width: auto;
            margin-right: 15px;
        }
        input[type="submit"] {
            background-color: #3498db;
            color: white;
            padding: 10px 30px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
        }
        input[type="submit"]:hover {
            background-color: #2980b9;
        }
        .result {
            background-color: #2ecc71;
            color: white;
            padding: 15px;
            border-radius: 8px;
            margin-top: 20px;
            text-align: center;
        }
        .result h3 {
            margin: 0 0 10px 0;
        }
        .result p {
            font-size: 28px;
            margin: 0;
            font-weight: bold;
        }
        hr {
            margin: 30px 0;
        }
        img {
            max-width: 100%;
            border-radius: 8px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
    </style>
</head>
<body>
    <div class="container">
        <h1> Medical Cost Predictor</h1>
        
        <div class="stats">
            <div class="stat-card">
                <h3>Accuracy (R²)</h3>
                <p>{{ accuracy }}%</p>
            </div>
            <div class="stat-card">
                <h3>Total Rows</h3>
                <p>{{ total }}</p>
            </div>
            <div class="stat-card">
                <h3>Total Columns</h3>
                <p>7</p>
            </div>
        </div>
        
        <hr>
        
        <h2>Predict Medical Cost</h2>
        <form method="POST">
            <div class="form-group">
                <label>Age:</label>
                <input type="number" name="age" min="18" max="100" required>
            </div>
            <div class="form-group">
                <label>BMI:</label>
                <input type="number" step="0.1" name="bmi" min="10" max="60" required>
            </div>
            <div class="form-group">
                <label>Children:</label>
                <input type="number" name="children" min="0" max="10" required>
            </div>
            <div class="form-group">
                <label>Smoker:</label>
                <div class="radio-group">
                    <label><input type="radio" name="smoker" value="yes" required> Yes</label>
                    <label><input type="radio" name="smoker" value="no" required> No</label>
                </div>
            </div>
            <div class="form-group">
                <input type="submit" value="Predict">
            </div>
        </form>
        
        {% if result %}
        <div class="result">
            <h3> Predicted Medical Cost</h3>
            <p>{{ result }}</p>
        </div>
        {% endif %}
        
        <hr>
        
        <h2>Sample Data (первые 10 строк)</h2>
        <table>
            <thead>
                <tr>
                    {% for col in columns[:6] %}
                    <th>{{ col }}</th>
                    {% endfor %}
                </tr>
            </thead>
            <tbody>
                {% for row in table_data[:10] %}
                <tr>
                    {% for col in columns[:6] %}
                    <td>{{ row[col] }}</td>
                    {% endfor %}
                </tr>
                {% endfor %}
            </tbody>
        </table>
        
        <hr>
        
        <h2>Model Graph</h2>
        <p><strong>X-Axis:</strong> Real Medical Cost</p>
        <p><strong>Y-Axis:</strong> Predicted Medical Cost</p>
        <img src="data:image/png;base64,{{ plot_url }}">
        
        <hr>
    </div>
</body>
</html>
"""
@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    
    data_plot = pd.read_csv('insurance.csv')
    data_plot['smoker'] = data_plot['smoker'].map({'yes': 1, 'no': 0})
    
    data_plot['bmi_smoker'] = data_plot['bmi'] * data_plot['smoker']
    data_plot['age_bmi'] = data_plot['age'] * data_plot['bmi']
    data_plot['age2'] = data_plot['age'] ** 2
    data_plot['bmi2'] = data_plot['bmi'] ** 2
    data_plot['children2'] = data_plot['children'] ** 2
    data_plot['age_smoker'] = data_plot['age'] * data_plot['smoker']
    
    X = data_plot[features]
    y = np.log1p(data_plot['charges'])
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    y_pred = model.predict(X_test)
    y_test_orig = np.expm1(y_test)
    y_pred_orig = np.expm1(y_pred)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test_orig, y_pred_orig, alpha=0.5)
    min_val = min(y_test_orig.min(), y_pred_orig.min())
    max_val = max(y_test_orig.max(), y_pred_orig.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    plt.xlabel('Реальные ($)')
    plt.ylabel('Предсказанные ($)')
    plt.grid(True, alpha=0.3)
    
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plot_url = base64.b64encode(buffer.getvalue()).decode()
    plt.close()
    
    accuracy = f"{model.score(X_test, y_test) * 100:.1f}"
    
    if request.method == 'POST':
        age = int(request.form['age'])
        bmi = float(request.form['bmi'])
        children = int(request.form['children'])
        smoker = request.form['smoker']
        
        smoker_val = 1 if smoker == 'yes' else 0
        input_data = pd.DataFrame([[
            age, bmi, children, smoker_val,
            bmi * smoker_val,
            age * bmi,
            age ** 2,
            bmi ** 2,
            children ** 2,
            age * smoker_val
        ]], columns=features)
        
        pred_log = model.predict(input_data)[0]
        prediction = np.expm1(pred_log)
        result = f"${prediction:,.2f}"
    
    table_data = data_display.head(20).to_dict('records')
    
    return render_template_string(
        HTML,
        plot_url=plot_url,
        table_data=table_data,
        columns=data_display.columns.tolist(),
        total=len(data_display),
        result=result,
        accuracy=accuracy
    )

if __name__ == '__main__':
    app.run(debug=True)