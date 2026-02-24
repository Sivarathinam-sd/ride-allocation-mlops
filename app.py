import gradio as gr
import pandas as pd
import numpy as np
import random
import pickle

with open('models/rf_regressor.pkl', 'rb') as f:
    regressor = pickle.load(f)


DATA_PATH = 'data/processed/cleaned_dataset.csv'
df = pd.read_csv(DATA_PATH)

def predict_random_record():
    random_index = random.randint(0, len(df) - 1)
    record = df.iloc[random_index]

    # Prepare input for prediction (drop reward if exists)
    input_df = pd.DataFrame([record.drop('reward', errors='ignore')])
    input_df = input_df.iloc[:, :].values
    prediction = regressor.predict(input_df)[0]

    # Format record as "column - value"
    formatted_record = "### Selected Record\n\n"
    for col in df.columns:
        formatted_record += f"- **{col}** - {record[col]:.2f}\n"

    # Format prediction output
    prediction_text = f"""
### 🔮 Prediction Result

**Predicted Reward:** {prediction:.4f}
"""

    return formatted_record, prediction_text


with gr.Blocks() as demo:

    gr.Markdown("# 🚖 Ride Allocation ML Prediction")
    gr.Markdown("Click the button to randomly select a record and predict reward.")

    btn = gr.Button("Select Random Record & Predict", variant="primary")

    record_output = gr.Markdown()
    prediction_output = gr.Markdown()

    btn.click(
        fn=predict_random_record,
        outputs=[record_output, prediction_output]
    )

if __name__ == "__main__":
    print("Hello from Ride ML app project")
    demo.launch(server_name='0.0.0.0', server_port=7860)