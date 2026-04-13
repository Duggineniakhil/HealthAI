"""
Quick script to inspect model architecture and test predictions.
Run: python inspect_model.py
"""
import os
import json
import numpy as np
from pathlib import Path
from PIL import Image
import io, sys

# Force UTF-8 output on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

def inspect_chexpert():
    try:
        import tensorflow as tf
        from tensorflow.keras.models import load_model

        model_path = MODELS_DIR / "xray_chexpert_multidisease_model.h5"
        labels_path = MODELS_DIR / "xray_chexpert_labels.json"

        print(f"\n{'='*60}")
        print("Loading CheXpert Multi-Disease Model...")
        model = load_model(str(model_path))

        with open(labels_path, "r") as f:
            labels = json.load(f)

        print(f"Label Count: {len(labels)}")
        print(f"Labels: {labels}")
        print(f"\nModel Input Shape: {model.input_shape}")
        print(f"Model Output Shape: {model.output_shape}")
        print(f"\nLast 5 layers:")
        for l in model.layers[-5:]:
            print(f"  - {l.name}: {type(l).__name__}")

        # Find last Conv2D (both direct and nested)
        last_conv = None
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv = layer.name
                break
        if not last_conv:
            for layer in model.layers:
                if hasattr(layer, 'layers'):
                    for sl in reversed(layer.layers):
                        if isinstance(sl, tf.keras.layers.Conv2D):
                            last_conv = sl.name
                            break
                    if last_conv:
                        break

        print(f"\nLast Conv2D Layer: {last_conv}")

        # Find the densenet layer name
        for layer in model.layers:
            if hasattr(layer, 'layers'):
                print(f"\nNested model layer: {layer.name}")
                # find all conv layers in it
                conv_layers = [sl.name for sl in layer.layers if isinstance(sl, tf.keras.layers.Conv2D)]
                print(f"  Conv layers count: {len(conv_layers)}")
                if conv_layers:
                    print(f"  Last conv in nested: {conv_layers[-1]}")
                break

        # Check output activation
        output_layer = model.layers[-1]
        cfg = output_layer.get_config()
        print(f"\nOutput Layer Name: {output_layer.name}")
        print(f"Output Activation: {cfg.get('activation', {})}")
        
        # Test with dummy images
        print(f"\n{'='*60}")
        print("Testing with dummy images...")
        for name, val in [("White 220", 220), ("Mid-Grey 128", 128), ("Dark 30", 30)]:
            dummy = np.full((1, 224, 224, 3), val / 255.0, dtype=np.float32)
            preds = model.predict(dummy, verbose=0)[0]
            result = {labels[i]: float(preds[i]) for i in range(len(labels))}
            sorted_result = sorted(result.items(), key=lambda x: x[1], reverse=True)
            print(f"\n[{name}]")
            for disease, prob in sorted_result:
                print(f"  {disease:20s}: {prob*100:.1f}%")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

def inspect_simple():
    try:
        import tensorflow as tf
        from tensorflow.keras.models import load_model

        model_path = MODELS_DIR / "xray_disease_model.h5"
        with open(MODELS_DIR / "xray_class_mapping.json") as f:
            class_map = {int(k): v for k, v in json.load(f).items()}

        print(f"\n{'='*60}")
        print("Loading Simple Binary Model...")
        model = load_model(str(model_path))
        print(f"Class Map: {class_map}")
        print(f"Input Shape: {model.input_shape}")
        print(f"Output Shape: {model.output_shape}")

        output_layer = model.layers[-1]
        print(f"Output Activation: {output_layer.get_config().get('activation', 'unknown')}")

        # Test grey and white images
        for name, val in [("White 220", 220), ("Mid-Grey 128", 128), ("Dark 30", 30)]:
            dummy = np.full((1, 224, 224, 3), val / 255.0, dtype=np.float32)
            pred = model.predict(dummy, verbose=0)[0][0]
            label = class_map.get(1 if pred >= 0.5 else 0)
            print(f"  [{name}] pred={pred:.4f} -> {label}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    inspect_chexpert()
    inspect_simple()
