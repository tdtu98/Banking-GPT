from huggingface_hub import snapshot_download
path = snapshot_download(repo_id="ai-lab/MBD-mini", repo_type="dataset", local_dir="mbd_mini")