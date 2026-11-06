"""Try the verifier on a few real files: genuine vs impostor claims."""
from config import FSDD_DIR
from verify_voice import VoiceVerifier

v = VoiceVerifier("svm")
print("Enrolled:", v.enrolled_ids, " threshold =", round(v.threshold, 3), "\n")

cases = [
    ("5_george_7.wav",   "george",  "genuine"),
    ("5_george_7.wav",   "lucas",   "impostor (known)"),
    ("2_theo_10.wav",    "george",  "impostor (unknown speaker)"),
    ("8_yweweler_3.wav", "nicolas", "impostor (unknown speaker)"),
]
for fname, claim, label in cases:
    r = v.verify_file(FSDD_DIR / fname, claim)
    print(f"{label:28s} claim={claim:8s} p={r['probability']:.3f} "
          f"-> {'ACCEPT' if r['accepted'] else 'REJECT'}")