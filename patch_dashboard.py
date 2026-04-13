"""
Patch dashboard/app.py for torchxrayvision backend compatibility.
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

path = "dashboard/app.py"
content = open(path, "r", encoding="utf-8").read()

changes = [
    # Footer note
    (
        'Baseline (No Finding): {no_finding_prob*100:.1f}% &nbsp;|&nbsp; Scores are calibrated relative to baseline',
        'Model: torchxrayvision DenseNet121 (CheXpert+NIH+MIMIC) &nbsp;|&nbsp; Direct sigmoid probabilities'
    ),
    # Confidence section heading
    (
        'DISEASE PROBABILITY (CALIBRATED)',
        'DISEASE PROBABILITY (18 CONDITIONS)'
    ),
    # Level thresholds (HIGH/MED/LOW for real probabilities)
    (
        'level = "HIGH" if prob > 0.7 else "MED" if prob > 0.35 else "LOW"',
        'level = "HIGH" if prob > 0.5 else "MED" if prob > 0.2 else "LOW"'
    ),
    # Verdict threshold - normal cutoff
    (
        'if is_normal or top_score < 0.15:',
        'if is_normal or top_score < 0.10:'
    ),
    # Verdict threshold - critical cutoff  
    (
        'elif top_score >= 0.7:',
        'elif top_score >= 0.5:'
    ),
    # Verdict threshold - suspected cutoff
    (
        'elif top_score >= 0.4:',
        'elif top_score >= 0.25:'
    ),
    # Severity classification
    (
        'sev = "Critical" if top_score >= 0.7 else "Moderate" if top_score >= 0.4 else "Possible"',
        'sev = "Critical" if top_score >= 0.5 else "Moderate" if top_score >= 0.25 else "Possible"'
    ),
    # Secondary indicators threshold
    (
        'if p >= 0.15:',
        'if p >= 0.08:'
    ),
    # AI rec threshold
    (
        'if top_score >= 0.7:',
        'if top_score >= 0.5:'
    ),
    # Chart top N diseases
    (
        'for k, v in sorted_preds[:6]',
        'for k, v in sorted_preds[:8]'
    ),
]

count = 0
not_found = []
for old, new in changes:
    if old in content:
        content = content.replace(old, new, 1)
        count += 1
        print(f"OK: {old[:55].strip()!r}")
    else:
        not_found.append(old)
        print(f"NOT FOUND: {old[:55].strip()!r}")

open(path, "w", encoding="utf-8").write(content)
print(f"\nApplied {count}/{len(changes)} patches.")
if not_found:
    print("Missing patches (may already be applied):")
    for s in not_found:
        print(f"  - {s[:70]!r}")
