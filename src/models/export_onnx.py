import os
import numpy as np
import onnx
import onnxruntime as rt
from xgboost import XGBClassifier
from onnxmltools import convert_xgboost
from onnxmltools.convert.common.data_types import FloatTensorType

def export_to_onnx():
    model_path = "models/model.json"
    onnx_output_path = "models/model.onnx"

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file missing at {model_path}. Run training first.")

    print(f"📦 [ONNX EXPORT] Loading native XGBoost model from {model_path}...")
    model = XGBClassifier()
    model.load_model(model_path)

    # Reset internal feature names so onnxmltools can map to generic indexed nodes
    model.get_booster().feature_names = None

    # 4 input features: transaction_amount, account_balance, credit_score, failed_transactions_24h
    initial_types = [("float_input", FloatTensorType([None, 4]))]

    print("🔄 [ONNX EXPORT] Converting computation graph to ONNX format...")
    onnx_model = convert_xgboost(
        model,
        initial_types=initial_types,
        target_opset=15
    )

    # Save to disk
    os.makedirs("models", exist_ok=True)
    onnx.save_model(onnx_model, onnx_output_path)
    print(f"✅ [ONNX EXPORT SUCCESS] Model saved to {onnx_output_path}")

    # Sanity Check with ONNX Runtime
    session = rt.InferenceSession(onnx_output_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    dummy_input = np.array([[120.50, 4500.0, 720.0, 0.0]], dtype=np.float32)
    
    outputs = session.run(None, {input_name: dummy_input})
    print(f"🧪 [ONNX SANITY CHECK] Prediction Probabilities: {outputs[1]}")

if __name__ == "__main__":
    export_to_onnx()