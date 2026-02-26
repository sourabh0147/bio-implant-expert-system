from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib

app = Flask(__name__)

# --- Load Assets ---
models = {}
wear_database = {}

def load_assets():
    global wear_database
    try:
        models['cof'] = joblib.load('random_forest_model.pkl')
        models['ocp'] = joblib.load('ocp_model.pkl')
        print("ML Models loaded.")
    except:
        print("Warning: ML models not found. Run backend script first.")

    try:
        wear_database = joblib.load('wear_database.pkl')
        print("Wear Database loaded.")
    except:
        print("Warning: 'wear_database.pkl' not found.")

load_assets()

# --- UPDATE: Standardized Alloy Names ---
VALID_ALLOY_TYPES = ['Pure Mg', 'Al-Mg-Bi', 'Al-Mg-Sr', 'Al-Mg-Zn']

def generate_expert_insight(cof, ocp, wear_depth):
    comments = []
    
    # Friction
    if cof < 0.20: comments.append("✅ **Tribology:** Low friction (Excellent).")
    elif cof > 0.40: comments.append("⚠️ **Tribology:** High friction (Risk of wear).")
    else: comments.append("ℹ️ **Tribology:** Moderate friction.")

    # Corrosion
    if ocp < -1.4: comments.append("⚠️ **Corrosion:** Highly active (Rapid degradation).")
    elif ocp > -1.25: comments.append("✅ **Corrosion:** More noble (Stable).")
    else: comments.append("ℹ️ **Corrosion:** Moderate activity.")

    # Wear
    if wear_depth == 'N/A':
        comments.append("❓ **Wear:** No data available for this alloy.")
    elif wear_depth > 20.0:
        comments.append(f"❌ **Wear:** Significant material loss ({wear_depth} µm).")
    elif wear_depth < 10.0:
        comments.append(f"✅ **Wear:** High wear resistance ({wear_depth} µm).")
    else:
        comments.append(f"ℹ️ **Wear:** Moderate material loss ({wear_depth} µm).")

    return comments

@app.route('/')
def index():
    return render_template('index.html', alloy_types=VALID_ALLOY_TYPES)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        alloy = data.get('alloyType')
        time = float(data.get('timestamp'))

        input_data = pd.DataFrame([[time, alloy]], columns=['Timestamp', 'Alloy_Type'])
        
        pred_cof = float(models['cof'].predict(input_data)[0]) if 'cof' in models else 0.0
        pred_ocp = float(models['ocp'].predict(input_data)[0]) if 'ocp' in models else 0.0

        wear_info = wear_database.get(alloy, {})
        wear_depth = wear_info.get('max_depth_um', 'N/A')

        insights = generate_expert_insight(pred_cof, pred_ocp, wear_depth)

        return jsonify({
            'predicted_cof': f"{pred_cof:.4f}",
            'predicted_ocp': f"{pred_ocp:.4f} V",
            'wear_metrics': wear_info,
            'comments': insights
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)