#!/usr/bin/env python3
"""Publish a compact, continuously refreshed GitHub Actions diagnostic snapshot."""
import base64
import datetime
import json
import os
import signal
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

REPO = os.environ["GITHUB_REPOSITORY"]
TOKEN = os.environ["GITHUB_TOKEN"]
BRANCH = os.environ.get("LIVE_DIAGNOSTICS_BRANCH", "live-diagnostics")
BASE_SHA = os.environ["GITHUB_SHA"]
RUN_ID = os.environ.get("GITHUB_RUN_ID", "unknown")
RUN_ATTEMPT = os.environ.get("GITHUB_RUN_ATTEMPT", "1")
API = "https://api.github.com"
PATH = "live-diagnostics/latest.txt"
FILES = ["live-status.txt", "emulator-startup.log", "gradle-instrumented-test.log", "emulator-logcat.txt"]
INTERVAL = 5
stopping = False

def stop_handler(_signum, _frame):
    global stopping
    stopping = True

signal.signal(signal.SIGTERM, stop_handler)
signal.signal(signal.SIGINT, stop_handler)

def api(method, path, payload=None):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        API + path,
        data=data,
        method=method,
        headers={
            "Authorization": "Bearer " + TOKEN,
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "tank2-live-diagnostics",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=20) as response:
        raw = response.read()
        return json.loads(raw.decode("utf-8")) if raw else {}

def ensure_branch():
    encoded = urllib.parse.quote(BRANCH, safe="")
    try:
        api("GET", f"/repos/{REPO}/git/ref/heads/{encoded}")
        print(f"LIVE_DIAGNOSTICS_BRANCH_READY branch={BRANCH}", flush=True)
    except urllib.error.HTTPError as error:
        if error.code != 404:
            raise
        api("POST", f"/repos/{REPO}/git/refs", {
            "ref": "refs/heads/" + BRANCH,
            "sha": BASE_SHA,
        })
        print(f"LIVE_DIAGNOSTICS_BRANCH_CREATED branch={BRANCH}", flush=True)

def snapshot():
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    chunks = [
        "TANK2 LIVE DIAGNOSTICS",
        f"updated_utc={now}",
        f"workflow_run_id={RUN_ID}",
        f"run_attempt={RUN_ATTEMPT}",
        f"branch={BRANCH}",
        "",
    ]
    for filename in FILES:
        chunks.append(f"===== {filename} (last 100 lines) =====")
        if os.path.isfile(filename):
            try:
                with open(filename, "r", encoding="utf-8", errors="replace") as stream:
                    lines = stream.readlines()[-100:]
                chunks.extend(line.rstrip("\n")[:1200] for line in lines)
            except Exception as error:
                chunks.append(f"READ_ERROR={type(error).__name__}: {error}")
        else:
            chunks.append("(not created yet)")
        chunks.append("")
    result = "\n".join(chunks) + "\n"
    return result[-90000:]

def publish():
    content = snapshot()
    encoded_branch = urllib.parse.quote(BRANCH, safe="")
    encoded_path = urllib.parse.quote(PATH, safe="/")
    current_sha = None
    try:
        existing = api("GET", f"/repos/{REPO}/contents/{encoded_path}?ref={encoded_branch}")
        current_sha = existing.get("sha")
    except urllib.error.HTTPError as error:
        if error.code != 404:
            raise
    payload = {
        "message": f"ci: refresh live diagnostics run {RUN_ID}",
        "content": base64.b64encode(content.encode("utf-8")).decode("ascii"),
        "branch": BRANCH,
    }
    if current_sha:
        payload["sha"] = current_sha
    api("PUT", f"/repos/{REPO}/contents/{encoded_path}", payload)
    print(f"LIVE_DIAGNOSTICS_PUBLISHED utc={datetime.datetime.now(datetime.timezone.utc).isoformat()} bytes={len(content.encode('utf-8'))}", flush=True)

def main():
    ensure_branch()
    while not stopping:
        started = time.monotonic()
        try:
            publish()
        except Exception as error:
            print(f"LIVE_DIAGNOSTICS_PUBLISH_ERROR type={type(error).__name__} detail={error}", file=sys.stderr, flush=True)
        remaining = max(0.0, INTERVAL - (time.monotonic() - started))
        deadline = time.monotonic() + remaining
        while not stopping and time.monotonic() < deadline:
            time.sleep(min(0.25, deadline - time.monotonic()))
    try:
        publish()
    except Exception as error:
        print(f"LIVE_DIAGNOSTICS_FINAL_PUBLISH_ERROR type={type(error).__name__} detail={error}", file=sys.stderr, flush=True)

if __name__ == "__main__":
    main()
