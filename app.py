import os
import json
import numpy as np
from PIL import Image
from dotenv import load_dotenv

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    jsonify
)
from flask_sqlalchemy import SQLAlchemy
from flask_login import (
    LoginManager,
    UserMixin,
    login_user,
    login_required,
    logout_user,
    current_user
)
from flask_bcrypt import Bcrypt

# Import core clinical ML Pipeline functions
from src.ml_pipeline import (
    DISEASE_CONFIGS,
    predict_tabular,
    predict_image_dl,
    train_disease_model,
    evaluate_disease_model,
    get_model_evaluation_metrics
)

load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'revive_clinical_ai_secret_key_2026')
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('SQLALCHEMY_DATABASE_URI', 'sqlite:///revive.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
login_manager.login_message_category = 'info'


# --- User Authentication Model ---
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(60), nullable=False)


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


with app.app_context():
    db.create_all()


# --- Authentication Routes ---
@app.route("/register", methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '')

        if not username or not email or not password:
            flash('All registration fields are required.', 'danger')
            return redirect(url_for('register'))

        user_exists = User.query.filter_by(username=username).first()
        email_exists = User.query.filter_by(email=email).first()
        if user_exists:
            flash('Username already exists. Please choose a different one.', 'danger')
            return redirect(url_for('register'))
        if email_exists:
            flash('Email already registered. Please login.', 'danger')
            return redirect(url_for('register'))

        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
        user = User(username=username, email=email, password=hashed_password)
        db.session.add(user)
        db.session.commit()
        flash('Your account has been created! You can now log in.', 'success')
        return redirect(url_for('login'))
    return render_template('register.html')


@app.route("/login", methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        user = User.query.filter_by(username=username).first()
        if user and bcrypt.check_password_hash(user.password, password):
            login_user(user)
            next_page = request.args.get('next')
            return redirect(next_page) if next_page else redirect(url_for('home'))
        else:
            flash('Login Unsuccessful. Please check username and password.', 'danger')
    return render_template('login.html')


@app.route("/logout")
def logout():
    logout_user()
    return redirect(url_for('index'))


# --- Public Navigation & Auxiliary Template Routes ---
@app.route("/")
@app.route("/index")
def index():
    return render_template('index.html')


@app.route("/about-us")
def about_us():
    return render_template('about-us.html')


@app.route("/services")
def services():
    return render_template('services.html')


@app.route("/elements")
def elements():
    return render_template('elements.html')


@app.route("/contact")
def contact():
    return render_template('contact.html')


@app.route("/index-icons")
def index_icons():
    return render_template('index-icons.html')


@app.route("/blog")
@app.route("/single-blog")
def blog():
    return render_template('single-blog.html')


# --- Disease Predictor Form Interfaces ---
@app.route("/models")
@login_required
def home():
    return render_template('home.html')


@app.route("/diabetes", methods=['GET', 'POST'])
@login_required
def diabetesPage():
    return render_template('diabetes.html')


@app.route("/cancer", methods=['GET', 'POST'])
@login_required
def cancerPage():
    return render_template('breast_cancer.html')


@app.route("/heart", methods=['GET', 'POST'])
@login_required
def heartPage():
    return render_template('heart.html')


@app.route("/kidney", methods=['GET', 'POST'])
@login_required
def kidneyPage():
    return render_template('kidney.html')


@app.route("/liver", methods=['GET', 'POST'])
@login_required
def liverPage():
    return render_template('liver.html')


@app.route("/stroke", methods=['GET', 'POST'])
@login_required
def strokePage():
    return render_template('stroke.html')


@app.route("/malaria", methods=['GET', 'POST'])
@login_required
def malariaPage():
    return render_template('malaria.html')


@app.route("/pneumonia", methods=['GET', 'POST'])
@login_required
def pneumoniaPage():
    return render_template('pneumonia.html')


# --- Prediction Endpoints ---
@app.route("/predict", methods=['POST', 'GET'])
@login_required
def predictPage():
    if request.method == 'GET':
        return redirect(url_for('home'))

    form_dict = request.form.to_dict()
    disease_type = form_dict.get('disease_type', '').lower().strip()

    # Intelligent disease identification matching features, forms and field counts
    if disease_type in DISEASE_CONFIGS:
        disease_key = disease_type
    elif 'radius_mean' in form_dict or len(form_dict) == 26:
        disease_key = 'cancer'
    elif 'pregnancies' in form_dict or ('glucose' in form_dict and 'insulin' in form_dict):
        disease_key = 'diabetes'
    elif 'chol' in form_dict or 'trestbps' in form_dict or 'thalach' in form_dict or len(form_dict) == 13:
        disease_key = 'heart'
    elif 'sg' in form_dict or 'bgr' in form_dict or 'rbc' in form_dict or len(form_dict) == 18:
        disease_key = 'kidney'
    elif any(k.lower() in ['total_bilirubin', 'alkaline_phosphotase', 'albumin'] for k in form_dict) or len(form_dict) == 10:
        disease_key = 'liver'
    elif 'avg_glucose_level' in form_dict or 'smoking_status' in form_dict or 'ever_married' in form_dict:
        disease_key = 'stroke'
    else:
        # Fallback length check
        count = len(form_dict)
        if count == 8:
            disease_key = 'diabetes'
        elif count == 26:
            disease_key = 'cancer'
        elif count == 13:
            disease_key = 'heart'
        elif count == 18:
            disease_key = 'kidney'
        elif count == 10:
            disease_key = 'liver'
        else:
            disease_key = 'stroke'

    page_map = {
        'diabetes': 'diabetesPage',
        'cancer': 'cancerPage',
        'heart': 'heartPage',
        'kidney': 'kidneyPage',
        'liver': 'liverPage',
        'stroke': 'strokePage'
    }
    return_endpoint = page_map.get(disease_key, 'home')

    try:
        result = predict_tabular(disease_key, form_dict)
        return render_template(
            'predict.html',
            pred=result['prediction'],
            result=result,
            disease_name=result['display_name'],
            return_url=url_for(return_endpoint)
        )
    except Exception as e:
        flash(f"Error processing assessment: {str(e)}", "danger")
        return redirect(url_for(return_endpoint))


@app.route("/malariapredict", methods=['POST', 'GET'])
@login_required
def malariapredictPage():
    if request.method == 'GET':
        return redirect(url_for('malariaPage'))

    try:
        if 'image' not in request.files or not request.files['image'].filename:
            return render_template('malaria.html', message="Please upload a microscopic blood smear cell image (.png, .jpg).")

        image_file = request.files['image']
        result = predict_image_dl('malaria', image_file)
        return render_template('malaria_predict.html', pred=result['prediction'], result=result)
    except Exception as e:
        return render_template('malaria.html', message=f"Prediction error: {str(e)}")


@app.route("/pneumoniapredict", methods=['POST', 'GET'])
@login_required
def pneumoniapredictPage():
    if request.method == 'GET':
        return redirect(url_for('pneumoniaPage'))

    try:
        if 'image' not in request.files or not request.files['image'].filename:
            return render_template('pneumonia.html', message="Please upload a chest X-Ray radiograph image (.png, .jpg).")

        image_file = request.files['image']
        result = predict_image_dl('pneumonia', image_file)
        return render_template('pneumonia_predict.html', pred=result['prediction'], result=result)
    except Exception as e:
        return render_template('pneumonia.html', message=f"Prediction error: {str(e)}")


# --- Interactive ML Pipeline Modules (Jupyter Notebook Themed) ---
@app.route("/training", methods=['GET', 'POST'])
@login_required
def training_page():
    selected_disease = "diabetes"
    result = None

    if request.method == 'POST':
        selected_disease = request.form.get("disease_key", "diabetes")
        try:
            test_size = float(request.form.get("test_size", 0.2))
        except ValueError:
            test_size = 0.2
        try:
            n_estimators = int(request.form.get("n_estimators", 100))
        except ValueError:
            n_estimators = 100

        try:
            train_output = train_disease_model(
                selected_disease,
                test_size=test_size,
                n_estimators=n_estimators
            )
            metrics_dict = train_output.get('metrics', train_output)
            dataset_name = metrics_dict.get('dataset_file', DISEASE_CONFIGS[selected_disease].get('dataset_file', 'dataset'))
            result = {
                "success": True,
                "message": f"Successfully re-trained {DISEASE_CONFIGS[selected_disease]['display_name']} on {dataset_name}!",
                "metrics": metrics_dict
            }
        except Exception as e:
            result = {
                "success": False,
                "message": f"Training error: {str(e)}",
                "metrics": None
            }

    return render_template(
        'training.html',
        configs=DISEASE_CONFIGS,
        selected_disease=selected_disease,
        result=result
    )


@app.route("/testing", methods=['GET', 'POST'])
@login_required
def testing_page():
    selected_disease = "diabetes"
    if request.method == 'POST':
        selected_disease = request.form.get("disease_key", "diabetes")

    try:
        eval_result = evaluate_disease_model(selected_disease)
    except Exception as e:
        flash(f"Error evaluating {selected_disease}: {str(e)}", "danger")
        eval_result = None

    return render_template(
        'testing.html',
        configs=DISEASE_CONFIGS,
        selected_disease=selected_disease,
        eval_result=eval_result
    )


@app.route("/evaluation", methods=['GET'])
@login_required
def evaluation_page():
    metrics = get_model_evaluation_metrics()
    return render_template('evaluation.html', metrics=metrics)


# --- REST API Endpoints for Programmatic ML Verification & Integration ---
@app.route("/api/metrics", methods=['GET'])
def api_metrics():
    """Returns verified evaluation metrics across all 8 disease models in JSON."""
    return jsonify(get_model_evaluation_metrics())


@app.route("/api/predict/<disease_key>", methods=['POST'])
@login_required
def api_predict(disease_key):
    """Programmatic inference endpoint for tabular and CNN models."""
    if disease_key not in DISEASE_CONFIGS:
        return jsonify({"status": "error", "message": f"Disease '{disease_key}' not recognized."}), 400

    cfg = DISEASE_CONFIGS[disease_key]
    try:
        if cfg["type"] == "tabular":
            data = request.get_json(silent=True) or request.form.to_dict()
            result = predict_tabular(disease_key, data)
            return jsonify(result)
        elif cfg["type"] == "cnn_image":
            if 'image' not in request.files:
                return jsonify({"status": "error", "message": "No image file provided in multipart/form-data."}), 400
            result = predict_image_dl(disease_key, request.files['image'])
            return jsonify(result)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


@app.route("/api/train/<disease_key>", methods=['POST'])
@login_required
def api_train(disease_key):
    """Triggers verified model retraining via API."""
    if disease_key not in DISEASE_CONFIGS:
        return jsonify({"status": "error", "message": f"Disease '{disease_key}' not recognized."}), 400

    data = request.get_json(silent=True) or {}
    test_size = float(data.get("test_size", 0.2))
    n_estimators = int(data.get("n_estimators", 100))

    try:
        metrics = train_disease_model(disease_key, test_size=test_size, n_estimators=n_estimators)
        return jsonify({"status": "success", "metrics": metrics})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True)