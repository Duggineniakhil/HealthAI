import tensorflow as tf
import numpy as np
import os

model_path = 'models/xray_disease_model.h5'
model = tf.keras.models.load_model(model_path)

print(f"{'Value':<10} | {'Normalization':<15} | {'Prediction':<10}")
print("-" * 40)

for val in [0.0, 127.0, 255.0]:
    # Test raw 0-255
    dummy_raw = np.full((1, 224, 224, 3), val, dtype=np.float32)
    p_raw = model.predict(dummy_raw, verbose=0)[0][0]
    print(f"{val:<10.1f} | {'None (0-255)':<15} | {p_raw:<10.4f}")

    # Test [0, 1]
    dummy_01 = np.full((1, 224, 224, 3), val / 255.0, dtype=np.float32)
    p_01 = model.predict(dummy_01, verbose=0)[0][0]
    print(f"{val:<10.1f} | {'0-1':<15} | {p_01:<10.4f}")

    # Test [-1, 1] (MobileNetV2 style)
    dummy_m11 = (val / 127.5) - 1.0
    dummy_m11 = np.full((1, 224, 224, 3), dummy_m11, dtype=np.float32)
    p_m11 = model.predict(dummy_m11, verbose=0)[0][0]
    print(f"{val:<10.1f} | {'-1 to 1':<15} | {p_m11:<10.4f}")
    
    # Test ImageNet (Mean subtraction)
    # Approx mean: [123.68, 116.779, 103.939]
    dummy_imgnet = val - 117.0
    dummy_imgnet = np.full((1, 224, 224, 3), dummy_imgnet, dtype=np.float32)
    p_imgnet = model.predict(dummy_imgnet, verbose=0)[0][0]
    print(f"{val:<10.1f} | {'Mean Sub':<15} | {p_imgnet:<10.4f}")
    print("-" * 40)
