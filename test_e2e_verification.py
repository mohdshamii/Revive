import io
import os
import sys

workspace = os.path.abspath(os.path.dirname(__file__))
if workspace not in sys.path:
    sys.path.insert(0, workspace)

import numpy as np
from PIL import Image
from app import app, db, User

def run_tests():
    print("=" * 70)
    print("STARTING REVIVE FULL END-TO-END VERIFICATION SUITE")
    print("=" * 70)

    client = app.test_client()
    passed = 0
    failed = 0

    def assert_test(condition, test_name):
        nonlocal passed, failed
        if condition:
            print(f" [PASS] {test_name}")
            passed += 1
        else:
            print(f" [FAIL] {test_name}")
            failed += 1

    # 1. Public Routes
    print("\n--- 1. Testing Public & Static Routes ---")
    public_urls = [
        ('/', "Root Landing Page"),
        ('/index', "Index Page"),
        ('/about-us', "About Us Page"),
        ('/services', "Services Catalog"),
        ('/elements', "Design Elements Page"),
        ('/contact', "Contact Page"),
        ('/index-icons', "Icon Gallery"),
        ('/blog', "Blog Page"),
        ('/single-blog', "Single Blog Article"),
        ('/register', "Registration Page"),
        ('/login', "Login Page"),
        ('/api/metrics', "Public API Metrics Endpoint")
    ]

    for url, desc in public_urls:
        resp = client.get(url)
        assert_test(resp.status_code == 200, f"{desc} ({url}) returns 200")

    # 2. Authentication Flow
    print("\n--- 2. Testing Authentication Flow ---")
    with app.app_context():
        # Ensure a test user exists
        test_user = User.query.filter_by(username="e2e_tester").first()
        if not test_user:
            from app import bcrypt
            test_user = User(
                username="e2e_tester",
                email="tester@revive.ai",
                password=bcrypt.generate_password_hash("Secret123!").decode('utf-8')
            )
            db.session.add(test_user)
            db.session.commit()

    # Login
    login_resp = client.post('/login', data={
        'username': 'e2e_tester',
        'password': 'Secret123!'
    }, follow_redirects=True)
    assert_test(login_resp.status_code == 200 and b'Revive' in login_resp.data, "Login successful with session cookie")

    # 3. Authenticated Views
    print("\n--- 3. Testing Authenticated Core Pages ---")
    auth_urls = [
        ('/models', "Models Dashboard", b"Revive"),
        ('/diabetes', "Diabetes Predictor Form", b"Pregnancies"),
        ('/cancer', "Breast Cancer Predictor Form", b"radius_mean"),
        ('/heart', "Heart Disease Predictor Form", b"age"),
        ('/kidney', "Kidney Disease Predictor Form", b"Blood Pressure"),
        ('/liver', "Liver Disease Predictor Form", b"Bilirubin"),
        ('/stroke', "Brain Stroke Predictor Form", b"Stroke"),
        ('/malaria', "Malaria Image Upload Page", b"Malaria"),
        ('/pneumonia', "Pneumonia Image Upload Page", b"Pneumonia"),
        ('/training', "Interactive Training Jupyter Notebook", b"Run Training Cell"),
        ('/testing', "Interactive Testing Jupyter Notebook", b"Holdout"),
        ('/evaluation', "Model Evaluation & Diagnostics Dashboard", b"Confusion Matrices")
    ]

    for url, desc, pattern in auth_urls:
        resp = client.get(url)
        assert_test(resp.status_code == 200 and pattern in resp.data, f"{desc} ({url}) rendered successfully")

    # 4. Tabular Model Predictions
    print("\n--- 4. Testing Tabular Model Predictions with Clinical Data ---")
    # Diabetes
    resp = client.post('/predict', data={
        'pregnancies': '6',
        'glucose': '148',
        'bloodpressure': '72',
        'skinthickness': '35',
        'insulin': '0',
        'bmi': '33.6',
        'diabetespedigreefunction': '0.627',
        'age': '50'
    })
    assert_test(resp.status_code == 200 and b"AI Diagnostic Prediction Report" in resp.data, "Diabetes prediction processed successfully")

    # Breast Cancer (26 features)
    cancer_data = {
        'radius_mean': '17.99', 'texture_mean': '10.38', 'perimeter_mean': '122.8', 'area_mean': '1001.0',
        'smoothness_mean': '0.1184', 'compactness_mean': '0.2776', 'concavity_mean': '0.3001', 'concave points_mean': '0.1471',
        'symmetry_mean': '0.2419', 'radius_se': '1.095', 'perimeter_se': '8.589', 'area_se': '153.4',
        'compactness_se': '0.04904', 'concavity_se': '0.05373', 'concave points_se': '0.01587', 'fractal_dimension_se': '0.006193',
        'radius_worst': '25.38', 'texture_worst': '17.33', 'perimeter_worst': '184.6', 'area_worst': '2019.0',
        'smoothness_worst': '0.1622', 'compactness_worst': '0.6656', 'concavity_worst': '0.7119', 'concave points_worst': '0.2654',
        'symmetry_worst': '0.4601', 'fractal_dimension_worst': '0.1189'
    }
    resp = client.post('/predict', data=cancer_data)
    assert_test(resp.status_code == 200 and b"AI Diagnostic Prediction Report" in resp.data, "Breast Cancer prediction processed successfully")

    # Heart Disease (13 features)
    heart_data = {
        'age': '63', 'sex': '1', 'cp': '3', 'trestbps': '145', 'chol': '233', 'fbs': '1',
        'restecg': '0', 'thalach': '150', 'exang': '0', 'oldpeak': '2.3', 'slope': '0', 'ca': '0', 'thal': '1'
    }
    resp = client.post('/predict', data=heart_data)
    assert_test(resp.status_code == 200 and b"AI Diagnostic Prediction Report" in resp.data, "Heart Disease prediction processed successfully")

    # Kidney Disease (18 features with string categories)
    kidney_data = {
        'age': '48', 'bp': '80', 'sg': '1.020', 'al': '1', 'su': '0',
        'rbc': 'normal', 'pc': 'normal', 'pcc': 'notpresent', 'ba': 'notpresent',
        'bgr': '121', 'bu': '36', 'sc': '1.2', 'pot': '4.6', 'wc': '7800',
        'htn': 'yes', 'dm': 'yes', 'cad': 'no', 'pe': 'no'
    }
    resp = client.post('/predict', data=kidney_data)
    assert_test(resp.status_code == 200 and b"AI Diagnostic Prediction Report" in resp.data, "Kidney Disease prediction with categorical strings processed")

    # Liver Disease (10 features)
    liver_data = {
        'Age': '65', 'Total_Bilirubin': '0.7', 'Direct_Bilirubin': '0.1',
        'Alkaline_Phosphotase': '187', 'Alamine_Aminotransferase': '16',
        'Aspartate_Aminotransferase': '18', 'Total_Protiens': '6.8',
        'Albumin': '3.3', 'Albumin_and_Globulin_Ratio': '0.9', 'Gender_Male': '1'
    }
    resp = client.post('/predict', data=liver_data)
    assert_test(resp.status_code == 200 and b"AI Diagnostic Prediction Report" in resp.data, "Liver Disease prediction processed")

    # Brain Stroke (16 features or form inputs)
    stroke_data = {
        'disease_type': 'stroke',
        'age': '67',
        'gender': 'Male',
        'ever_married': 'Yes',
        'hypertension': '1',
        'heart_disease': '0',
        'avg_glucose_level': '228.69',
        'bmi': '36.6',
        'work_type': 'Private',
        'Residence_type': 'Urban',
        'smoking_status': 'formerly smoked'
    }
    resp = client.post('/predict', data=stroke_data)
    assert_test(resp.status_code == 200 and b"AI Diagnostic Prediction Report" in resp.data, "Brain Stroke prediction processed")

    # 5. Deep Learning Image Predictions
    print("\n--- 5. Testing Deep Learning CNN Predictions ---")
    # Synthetic 36x36 RGB image for Malaria
    img_rgb = Image.new('RGB', (36, 36), color=(180, 50, 100))
    buf_rgb = io.BytesIO()
    img_rgb.save(buf_rgb, format='PNG')
    buf_rgb.seek(0)

    resp = client.post('/malariapredict', data={'image': (buf_rgb, 'cell.png')}, content_type='multipart/form-data')
    assert_test(resp.status_code == 200 and b"Malaria Cell Image Diagnosis" in resp.data, "Malaria CNN image inference successful")

    # Synthetic 36x36 Grayscale image for Pneumonia
    img_gray = Image.new('L', (36, 36), color=128)
    buf_gray = io.BytesIO()
    img_gray.save(buf_gray, format='PNG')
    buf_gray.seek(0)

    resp = client.post('/pneumoniapredict', data={'image': (buf_gray, 'xray.png')}, content_type='multipart/form-data')
    assert_test(resp.status_code == 200 and b"Chest X-Ray Pneumonia Assessment" in resp.data, "Pneumonia CNN radiograph inference successful")

    # 6. ML Pipeline Interactive Training, Testing, Evaluation
    print("\n--- 6. Testing Interactive Jupyter Notebook Pipeline Modules ---")
    # Training POST
    resp = client.post('/training', data={
        'disease_key': 'diabetes',
        'test_size': '0.2',
        'n_estimators': '50'
    })
    assert_test(resp.status_code == 200 and b"Successfully re-trained" in resp.data, "Model Training Cell execution and re-serialization")

    # Testing POST
    resp = client.post('/testing', data={'disease_key': 'heart'})
    assert_test(resp.status_code == 200 and b"Test Suite Execution Completed" in resp.data, "Model Testing Suite execution with real test metrics")

    # Evaluation GET
    resp = client.get('/evaluation')
    assert_test(resp.status_code == 200 and b"Model Performance Comparison Table" in resp.data, "Evaluation Metrics Dashboard with full telemetry")

    # 7. REST APIs
    print("\n--- 7. Testing REST APIs ---")
    resp = client.get('/api/metrics')
    data = resp.get_json()
    assert_test(resp.status_code == 200 and len(data) >= 8, f"API Metrics returns all {len(data)} models")

    resp = client.post('/api/predict/diabetes', json={
        'pregnancies': 2, 'glucose': 120, 'bloodpressure': 70,
        'skinthickness': 25, 'insulin': 80, 'bmi': 26.5,
        'diabetespedigreefunction': 0.45, 'age': 32
    })
    api_res = resp.get_json()
    assert_test(resp.status_code == 200 and api_res.get('status') == 'success', f"API Programmatic Prediction: label={api_res.get('label')}")

    print("\n" + "=" * 70)
    print(f"FINAL VERIFICATION SUMMARY: {passed} PASSED, {failed} FAILED")
    print("=" * 70)

    if failed > 0:
        sys.exit(1)

if __name__ == "__main__":
    run_tests()
