import subprocess
import sys

if __name__ == "__main__":
    print("======================================================================")
    print("  [08] LAUNCHING MULTIMODAL BIOMETRIC STREAMLIT DASHBOARD")
    print("======================================================================")
    cmd = [sys.executable, "-m", "streamlit", "run", "app.py"]
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n[08] Streamlit dashboard stopped by user.")
