"""Downloads the three Hugging Face models into ./models once (about 4.5 GB), so the demo later runs without internet."""
import getpass, os, sys
from huggingface_hub import snapshot_download
from huggingface_hub.utils import GatedRepoError, HfHubHTTPError

HERE = os.path.dirname(os.path.abspath(__file__))
SKIP = ["*.bin", "*.h5", "*.msgpack", "*.ot", "onnx/*", "openvino/*", "*.onnx"]
MODELS = [("distilbert/distilbert-base-uncased", "distilbert-base-uncased", False),
          ("sentence-transformers/all-MiniLM-L6-v2", "all-MiniLM-L6-v2", False),
          ("ai4bharat/indictrans2-indic-en-1B", "indictrans2-indic-en-1B", True)]
token = os.environ.get("HF_TOKEN")
for repo, name, gated in MODELS:
    dest = os.path.join(HERE, "models", name)
    if os.path.exists(os.path.join(dest, ".done")):
        print("already downloaded:", name); continue
    if gated and not token:
        token = getpass.getpass("Paste your Hugging Face token (it starts with hf_; typing is hidden) and press Enter: ").strip()
    print(f"downloading {repo} ...", flush=True)
    try:
        snapshot_download(repo, local_dir=dest, ignore_patterns=SKIP, token=token if gated else None)
    except (GatedRepoError, HfHubHTTPError) as e:
        if gated:
            sys.exit(f"\nHugging Face refused {repo}. Check that you are logged in at huggingface.co, that the agree button on "
                     f"https://huggingface.co/{repo} was clicked, and that the token is right. Then run setup.sh again.\n({e})")
        raise
    open(os.path.join(dest, ".done"), "w").write("ok")
print("all models downloaded")
