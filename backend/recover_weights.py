import torchxrayvision as xrv
import torch
import os
import sys

def download_and_verify():
    print("🚀 Initializing torchxrayvision weight recovery...")
    try:
        # This will trigger download if missing
        print("📥 Requesting DenseNet121 (all) weights...")
        model = xrv.models.DenseNet(weights="densenet121-res224-all")
        
        # Verify
        pathologies = model.pathologies
        print(f"✅ Success! Model loaded with {len(pathologies)} pathologies.")
        print(f"Pathologies Sample: {pathologies[:5]}")
        
        # Test trace to ensure it's functional
        test_input = torch.zeros(1, 1, 224, 224)
        with torch.no_grad():
            output = model(test_input)
            print(f"✅ Model inference verification: OK (Output shape: {output.shape})")
            
        return True
    except Exception as e:
        print(f"❌ Weight recovery failed: {e}")
        return False

if __name__ == "__main__":
    success = download_and_verify()
    if not success:
        sys.exit(1)
    print("✨ Weight recovery complete.")
