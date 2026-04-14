import tensorflow as tf
import numpy as np
import json

model_path = 'models/xray_chexpert_multidisease_model.h5'
labels_path = 'models/xray_chexpert_labels.json'

model = tf.keras.models.load_model(model_path)
with open(labels_path, 'r') as f:
    labels = json.load(f)

print(f"{'Value':<10} | {'Normalization':<15} | {'Top Label':<20} | {'Prob':<10}")
print("-" * 65)

for val in [0.0, 127.0, 255.0]:
    norms = [
        ("None (0-255)", lambda x: x),
        ("0-1", lambda x: x / 255.0),
        ("-1 to 1", lambda x: (x / 127.5) - 1.0),
        ("DenseNet", lambda x: tf.keras.applications.densenet.preprocess_input(x))
    ]
    
    for name, func in norms:
        dummy = np.full((1, 224, 224, 3), val, dtype=np.float32)
        dummy = func(dummy)
        preds = model.predict(dummy, verbose=0)[0]
        top_idx = np.argmax(preds)
        print(f"{val:<10.1f} | {name:<15} | {labels[top_idx]:<20} | {preds[top_idx]:<10.4f}")
    print("-" * 65)
