
import subprocess, sys

def main():
    code = subprocess.call([sys.executable, "-m", "src.analysis"])
    if code != 0:
        raise SystemExit(code)

if __name__ == "__main__":
    main()
