import gradio as gr
import pandas as pd
import joblib
import numpy as np

# --- 1. Load Model and Scaler ---

# NOTE: The model and scaler must be saved using the updated 'real_time_prediction.py' script first.
try:
    LIVER_MODEL = joblib.load(r'E:\AI in Medical Diagnosis\Liver Disease Detection\saved models\liver_model.pkl')
    LIVER_SCALER = joblib.load(r'E:\AI in Medical Diagnosis\Liver Disease Detection\saved models\liver_scaler.pkl')
except FileNotFoundError:
    print("Error: Model or Scaler file not found.")
    print("Please run 'real_time_prediction.py' first to generate 'liver_model.pkl' and 'liver_scaler.pkl'.")
    LIVER_MODEL = None
    LIVER_SCALER = None

# Feature order (must match the order used during training)
FEATURE_ORDER = [
    'Age', 'Total_Bilirubin', 'Direct_Bilirubin', 'Alkaline_Phosphotase',
    'Alamine_Aminotransferase', 'Aspartate_Aminotransferase',
    'Total_Protiens', 'Albumin', 'Albumin_and_Globulin_Ratio'
]


# --- 2. Prediction Function for Gradio ---

def diagnose_patient(
    age, total_bilirubin, direct_bilirubin, alkaline_phosphotase, 
    alamine_aminotransferase, aspartate_aminotransferase, 
    total_protiens, albumin, albumin_and_globulin_ratio
):
    """
    Predicts liver disease status from raw patient inputs.
    """
    if LIVER_MODEL is None or LIVER_SCALER is None:
        return "MODEL NOT LOADED", "Model files missing. Check console for error."

    # 1. Collect inputs into a single list
    raw_data = [
        age, total_bilirubin, direct_bilirubin, alkaline_phosphotase,
        alamine_aminotransferase, aspartate_aminotransferase,
        total_protiens, albumin, albumin_and_globulin_ratio
    ]
    
    # 2. Convert to DataFrame (ensuring correct feature order)
    input_df = pd.DataFrame([raw_data], columns=FEATURE_ORDER)

    # 3. Scale the data using the *fitted* scaler
    scaled_data = LIVER_SCALER.transform(input_df)

    # 4. Get Prediction and Probability
    prediction = LIVER_MODEL.predict(scaled_data)[0]
    probability = LIVER_MODEL.predict_proba(scaled_data)[0][1]

    # 5. Format Output
    prob_percent = f"{probability * 100:.2f}%"

    if prediction == 1:
        # High Sensitivity Model: Flagging risk is the priority
        diagnosis_text = "RISK DETECTED: Liver Disease HIGH Probability"
        color = "red"
        interpretation = (
            f"The model predicts a high risk of Liver Disease (Probability: {prob_percent}). "
            "Given the model's high sensitivity (95.2% on the test set), this patient should be "
            "**flagged for immediate clinical review and further diagnostic testing** to prevent a missed diagnosis."
        )
    else:
        diagnosis_text = "Low Probability of Liver Disease"
        color = "green"
        interpretation = (
            f"The model predicts a low risk of Liver Disease (Probability: {prob_percent}). "
            "Continue routine monitoring. Note the model's low specificity suggests some healthy patients "
            "may be misclassified if their biomarkers are borderline."
        )

    # Use HTML/Markdown for better formatting in Gradio
    html_output = f"<h2 style='color: {color}; text-align: center;'>{diagnosis_text}</h2>"
    
    return html_output, interpretation

# --- 3. Define Gradio Interface Components ---

# Define the input components with clinically appropriate ranges and descriptions
input_components = [
    gr.Slider(minimum=1, maximum=100, step=1, value=45, label="Age (years)"),
    gr.Slider(minimum=0.2, maximum=75.0, step=0.1, value=1.0, label="Total Bilirubin (mg/dL)"),
    gr.Slider(minimum=0.1, maximum=35.0, step=0.1, value=0.2, label="Direct Bilirubin (mg/dL)"),
    gr.Slider(minimum=50, maximum=1500, step=1, value=180, label="Alkaline Phosphatase (IU/L)"),
    gr.Slider(minimum=5, maximum=500, step=1, value=25, label="ALT / Alamine Aminotransferase (IU/L)"),
    gr.Slider(minimum=5, maximum=500, step=1, value=30, label="AST / Aspartate Aminotransferase (IU/L)"),
    gr.Slider(minimum=2.0, maximum=10.0, step=0.1, value=7.0, label="Total Proteins (gm/dL)"),
    gr.Slider(minimum=1.0, maximum=6.0, step=0.1, value=3.5, label="Albumin (gm/dL)"),
    gr.Slider(minimum=0.1, maximum=3.0, step=0.01, value=1.0, label="Albumin and Globulin Ratio"),
]

# Define the output components
output_components = [
    gr.HTML(label="Prediction Result"),
    gr.Markdown(label="Clinical Interpretation and Confidence"),
]

# --- 4. Launch the Gradio App ---

iface = gr.Interface(
    fn=diagnose_patient,
    inputs=input_components,
    outputs=output_components,
    title="🔬 Early Liver Disease Risk Predictor (Logistic Regression Model)",
    description=(
        "Enter the patient's routine Liver Function Test (LFT) results below. "
        "The model uses a Logistic Regression classifier trained on the ILPD dataset "
        "to flag patients at high risk for liver disease."
    ),
    live=False,
    allow_flagging='never'
)

if __name__ == "__main__":
    iface.launch()