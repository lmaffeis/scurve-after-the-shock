from scurve.config import load_config
from scurve.external import build_all

if __name__ == "__main__":
    cfg = load_config()
    build_all(cfg["paths"]["external_dir"])
    print("external data written to", cfg["paths"]["external_dir"])
