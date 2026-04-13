"""
Patches the findings panel in dashboard/app.py to fix:
1. HTML code showing as raw text (indented multiline f-string triggers Streamlit code block)
2. Show primary diagnosis clearly with severity
"""
import re

path = "dashboard/app.py"
content = open(path, "r", encoding="utf-8").read()

# ---- Replace the findings section (lines ~490-510) ----
# We search by unique markers in the old code
OLD_MARKER_START = "# Compute logic colors for scan status"
OLD_MARKER_END   = '""", unsafe_allow_html=True)\n    \n                        # Line Chart'

start_idx = content.find(OLD_MARKER_START)
end_idx   = content.find(OLD_MARKER_END)

if start_idx == -1:
    print("ERROR: Could not find start marker. File may already be patched.")
    exit(1)
if end_idx == -1:
    print("ERROR: Could not find end marker.")
    exit(1)

# Include the end marker itself up to the \n after it
end_idx_full = end_idx + len(OLD_MARKER_END)

OLD_BLOCK = content[start_idx:end_idx_full]

NEW_BLOCK = '''# Compute logic colors for scan status
                        status_text_color = "#10b981" if "Validated" in st.session_state['scan_status'] else ("#f59e0b" if "Review" in st.session_state['scan_status'] else "#fff")

                        # Build severity info
                        if is_normal or top_score < 0.15:
                            primary_finding_html = (
                                "<span style='color:#10b981;font-size:20px;font-weight:800;'>&#9989; NORMAL SCAN</span>"
                                f"<br><span style='color:#8ba0b8;font-size:13px;'>No significant pathology detected. "
                                f"Normal confidence: {no_finding_prob*100:.0f}%</span>"
                            )
                            ai_rec = "<span style='color:#10b981;'>No severe anomalies. Routine follow-up recommended.</span>"
                        else:
                            sev = "Critical" if top_score >= 0.7 else "Moderate" if top_score >= 0.4 else "Possible"
                            sev_color = "#ef4444" if sev == "Critical" else ("#f59e0b" if sev == "Moderate" else "#00f0ff")
                            primary_finding_html = (
                                f"<span style='color:{sev_color};font-size:20px;font-weight:800;'>{highest_dis[0].upper()}</span>"
                                f"<br><span style='color:#8ba0b8;font-size:13px;'>Severity: <strong style='color:{sev_color};'>{sev}</strong>"
                                f" &nbsp;|&nbsp; Score: <strong style='color:#fff;'>{top_score*100:.1f}%</strong></span>"
                            )
                            if top_score >= 0.7:
                                ai_rec = "<span style='color:#ef4444;'>&#9888; Critical anomaly. Immediate clinical review required.</span>"
                            else:
                                ai_rec = "<span style='color:#f59e0b;'>Possible finding detected. Further evaluation advised.</span>"

                        secondary_html = ""
                        for d, p in sorted_preds[1:4]:
                            if p >= 0.15:
                                secondary_html += (
                                    f"<span style='display:inline-block;background:rgba(255,255,255,0.07);"
                                    f"border-radius:5px;padding:3px 8px;margin:3px 3px 3px 0;"
                                    f"color:#8ba0b8;font-size:11px;'>{d} ({p*100:.0f}%)</span>"
                                )
                        if not secondary_html:
                            secondary_html = "<span style='color:#8ba0b8;font-size:13px;'>None above threshold</span>"

                        # Findings panel - built as concatenated string to avoid Streamlit code-block bug
                        findings_html = (
                            "<div style='background:rgba(16,25,41,0.85);padding:24px;border-radius:16px;"
                            "border:1px solid rgba(43,62,88,0.5);min-height:400px;margin-bottom:24px;"
                            "overflow-y:auto;box-shadow:0 10px 30px rgba(0,0,0,0.3);'>"
                            "<strong style='color:#fff;font-size:11px;letter-spacing:1.5px;"
                            "display:block;margin-bottom:18px;'>FINDINGS &amp; NOTES</strong>"
                            "<div style='margin-bottom:16px;border-bottom:1px solid rgba(255,255,255,0.07);padding-bottom:16px;'>"
                            "<span style='color:#8ba0b8;font-size:10px;font-weight:700;letter-spacing:1px;'>PRIMARY DIAGNOSIS</span>"
                            "<br><br>" + primary_finding_html + "</div>"
                            "<div style='margin-bottom:16px;border-bottom:1px solid rgba(255,255,255,0.07);padding-bottom:16px;'>"
                            "<span style='color:#8ba0b8;font-size:10px;font-weight:700;letter-spacing:1px;'>SECONDARY INDICATORS</span>"
                            "<br><br>" + secondary_html + "</div>"
                            f"<div style='margin-bottom:16px;border-bottom:1px solid rgba(255,255,255,0.07);padding-bottom:16px;'>"
                            f"<span style='color:#8ba0b8;font-size:10px;font-weight:700;letter-spacing:1px;'>DR. REVIEW STATUS</span>"
                            f"<br><br><span style='color:{status_text_color};font-size:14px;font-weight:700;'>"
                            f"{st.session_state['scan_status']}</span></div>"
                            "<div><span style='color:#8ba0b8;font-size:10px;font-weight:700;letter-spacing:1px;'>AI RECOMMENDATION</span>"
                            "<br><br>" + ai_rec + "</div></div>"
                        )
                        findings_placeholder.markdown(findings_html, unsafe_allow_html=True)

                        # Line Chart'''

content = content[:start_idx] + NEW_BLOCK + content[end_idx_full:]

open(path, "w", encoding="utf-8").write(content)
print(f"SUCCESS: Patched findings panel in {path}")
print(f"  Old block was {len(OLD_BLOCK)} chars, new block is {len(NEW_BLOCK)} chars")
