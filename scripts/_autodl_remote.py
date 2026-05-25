"""One-shot helper: SSH to AutoDL, sync project + run training, download results.

Reads SSH credentials from env vars (so the password never lives in code/history):
    AUTODL_HOST   e.g. region-46.seetacloud.com
    AUTODL_PORT   e.g. 49007
    AUTODL_USER   default: root
    AUTODL_PASS   the instance password

Sub-commands:
    upload   - rsync-like sync local → remote /root/beverage_ai/
    setup    - run setup_autodl.sh on remote
    train    - run training (streams stdout back live)
    download - pull models/ + log back to local
    full     - upload + setup + train + download
    exec CMD - run arbitrary single command on remote
"""
from __future__ import annotations

import fnmatch
import os
import stat
import sys
import time
from pathlib import Path

import paramiko


def _connect() -> paramiko.SSHClient:
    host = os.environ.get("AUTODL_HOST")
    port = int(os.environ.get("AUTODL_PORT", "22"))
    user = os.environ.get("AUTODL_USER", "root")
    password = os.environ.get("AUTODL_PASS")
    if not (host and password):
        raise SystemExit("Set AUTODL_HOST + AUTODL_PASS env vars (and AUTODL_PORT)")
    print(f"connecting to {user}@{host}:{port} ...", flush=True)
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    cli.connect(hostname=host, port=port, username=user, password=password,
                look_for_keys=False, allow_agent=False, timeout=30,
                banner_timeout=30, auth_timeout=30)
    return cli


# ----- exclusion patterns matching scripts/upload_to_autodl.sh -----
EXCLUDE_DIRS = {
    ".venv", ".venv_hf", "__pycache__", ".pytest_cache", ".git",
    "node_modules", ".idea", ".vscode", ".hypothesis",
}
EXCLUDE_FILES = {"*.pyc", "*.pyo", "*.duckdb", ".DS_Store"}


def _walk_local(root: Path):
    """Yield (rel_path, full_path, is_dir) for everything we want to upload."""
    for dirpath, dirnames, filenames in os.walk(root):
        # prune
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]
        rel_dir = Path(dirpath).relative_to(root)
        if str(rel_dir) != ".":
            yield rel_dir, Path(dirpath), True
        for fname in filenames:
            if any(fnmatch.fnmatch(fname, pat) for pat in EXCLUDE_FILES):
                continue
            full = Path(dirpath) / fname
            yield (rel_dir / fname if str(rel_dir) != "." else Path(fname)), full, False


def _ensure_remote_dir(sftp: paramiko.SFTPClient, remote_path: str) -> None:
    """mkdir -p on remote via SFTP."""
    parts = remote_path.strip("/").split("/")
    cur = ""
    for p in parts:
        cur = cur + "/" + p
        try:
            sftp.stat(cur)
        except FileNotFoundError:
            sftp.mkdir(cur)


def upload(cli: paramiko.SSHClient, local_root: Path, remote_root: str) -> None:
    """Upload the project. Skips files that are unchanged (size+mtime check)."""
    sftp = cli.open_sftp()
    _ensure_remote_dir(sftp, remote_root)

    # Specific top-level entries we want to ship
    targets = [
        "beverage_ai", "scripts", "tests", "demo", "docs",
        "data/ingredients", "data/priors", "data/recipes", "data/reviews/raw",
        "pyproject.toml", "README.md", ".python-version", ".env.example",
    ]

    total = 0
    skipped = 0
    uploaded = 0
    for top in targets:
        local = local_root / top
        if not local.exists():
            print(f"  skip {top} (not present locally)")
            continue
        if local.is_file():
            total += 1
            remote_path = f"{remote_root}/{top}"
            _ensure_remote_dir(sftp, str(Path(remote_path).parent))
            if _needs_upload(sftp, local, remote_path):
                sftp.put(str(local), remote_path)
                uploaded += 1
                print(f"  ↑ {top}")
            else:
                skipped += 1
        else:
            for rel, full, is_dir in _walk_local(local):
                rel_in_remote = f"{top}/{rel.as_posix()}"
                remote_path = f"{remote_root}/{rel_in_remote}"
                if is_dir:
                    _ensure_remote_dir(sftp, remote_path)
                    continue
                total += 1
                _ensure_remote_dir(sftp, str(Path(remote_path).parent).replace("\\", "/"))
                if _needs_upload(sftp, full, remote_path):
                    sftp.put(str(full), remote_path)
                    uploaded += 1
                    if uploaded % 50 == 0:
                        print(f"  ↑ {uploaded} files so far ...", flush=True)
                else:
                    skipped += 1

    sftp.close()
    print(f"upload done: {uploaded} uploaded, {skipped} unchanged, {total} total files")


def _needs_upload(sftp: paramiko.SFTPClient, local_path: Path, remote_path: str) -> bool:
    try:
        st = sftp.stat(remote_path)
    except FileNotFoundError:
        return True
    local_size = local_path.stat().st_size
    return st.st_size != local_size


def stream_exec(cli: paramiko.SSHClient, command: str, env: dict | None = None) -> int:
    """Run a command remotely; stream stdout + stderr to local stdout. Return exit code."""
    if env:
        env_prefix = " ".join(f"{k}={v!r}" for k, v in env.items())
        command = f"{env_prefix} {command}"
    print(f"> {command}", flush=True)
    chan = cli.get_transport().open_session()
    chan.set_combine_stderr(True)
    chan.exec_command(command)
    while True:
        if chan.recv_ready():
            data = chan.recv(4096)
            if data:
                sys.stdout.buffer.write(data)
                sys.stdout.flush()
        if chan.exit_status_ready() and not chan.recv_ready():
            break
        time.sleep(0.05)
    while chan.recv_ready():
        sys.stdout.buffer.write(chan.recv(4096))
        sys.stdout.flush()
    return chan.recv_exit_status()


def download(cli: paramiko.SSHClient, remote_root: str, local_root: Path,
             paths: list[str]) -> None:
    sftp = cli.open_sftp()
    for relpath in paths:
        remote = f"{remote_root}/{relpath}"
        local = local_root / relpath
        try:
            st = sftp.stat(remote)
        except FileNotFoundError:
            print(f"  remote not found: {remote}")
            continue
        if stat.S_ISDIR(st.st_mode):
            for fname in sftp.listdir(remote):
                rem_f = f"{remote}/{fname}"
                loc_f = local / fname
                loc_f.parent.mkdir(parents=True, exist_ok=True)
                try:
                    sftp.get(rem_f, str(loc_f))
                    print(f"  ↓ {relpath}/{fname}")
                except Exception as e:
                    print(f"  failed {fname}: {e}")
        else:
            local.parent.mkdir(parents=True, exist_ok=True)
            sftp.get(remote, str(local))
            print(f"  ↓ {relpath}")
    sftp.close()


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    cmd = sys.argv[1]
    local_root = Path("D:/2026new/paper/beverage_ai")
    remote_root = "/root/beverage_ai"

    cli = _connect()
    try:
        if cmd in ("upload", "full"):
            upload(cli, local_root, remote_root)
        if cmd in ("setup", "full"):
            rc = stream_exec(cli, f"cd {remote_root} && bash scripts/setup_autodl.sh")
            if rc != 0:
                sys.exit(f"setup failed (rc={rc})")
        if cmd in ("train", "full"):
            epochs = os.environ.get("EPOCHS", "50")
            extra = os.environ.get("TRAIN_EXTRA", "")
            rc = stream_exec(
                cli,
                f"cd {remote_root} && source .venv/bin/activate && "
                f"python -u scripts/train_sensory_gnn_stage1.py "
                f"--epochs {epochs} --device auto --amp --patience 10 --tag autodl "
                f"{extra}",
            )
            if rc != 0:
                sys.exit(f"training failed (rc={rc})")
        if cmd in ("download", "full"):
            download(cli, remote_root, local_root, [
                "models/sensory_gnn_stage1_prototype.pt",
                "models/sensory_gnn_stage1_best.pt",
                "models/sensory_gnn_stage1_log.json",
            ])
        if cmd == "exec":
            if len(sys.argv) < 3:
                print("usage: exec <remote command>"); sys.exit(1)
            rc = stream_exec(cli, " ".join(sys.argv[2:]))
            sys.exit(rc)
    finally:
        cli.close()


if __name__ == "__main__":
    main()
