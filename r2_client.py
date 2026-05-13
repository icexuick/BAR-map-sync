"""
Cloudflare R2 client helper.

Reads credentials from .env, exposes a small upload API with content-type
detection and skip-if-unchanged semantics (compares local MD5 against the
remote ETag, which R2 sets to the MD5 for single-PUT uploads).

Required env vars:
    R2_ACCOUNT_ID
    R2_ACCESS_KEY_ID
    R2_SECRET_ACCESS_KEY
    R2_BUCKET
    R2_PUBLIC_BASE  (e.g. https://pub-<hash>.r2.dev — used only for log/build URLs)
"""

import hashlib
import mimetypes
import os
from typing import Optional

import boto3
from botocore.client import Config
from botocore.exceptions import ClientError
from dotenv import load_dotenv

load_dotenv()

R2_ACCOUNT_ID = os.environ.get("R2_ACCOUNT_ID")
R2_ACCESS_KEY_ID = os.environ.get("R2_ACCESS_KEY_ID")
R2_SECRET_ACCESS_KEY = os.environ.get("R2_SECRET_ACCESS_KEY")
R2_BUCKET = os.environ.get("R2_BUCKET")
R2_PUBLIC_BASE = (os.environ.get("R2_PUBLIC_BASE") or "").rstrip("/")

# Custom content-type overrides for extensions mimetypes doesn't know about
# or guesses wrong for our pipeline.
CONTENT_TYPE_OVERRIDES = {
    ".glb": "model/gltf-binary",
    ".gltf": "model/gltf+json",
    ".webp": "image/webp",
    ".hdr": "image/vnd.radiance",
    ".json": "application/json",
}


def _require(name: str, value: Optional[str]) -> str:
    if not value:
        raise SystemExit(f"Missing required env var: {name} (check your .env)")
    return value


def make_client():
    """Build an S3-compatible boto3 client pointing at R2."""
    _require("R2_ACCOUNT_ID", R2_ACCOUNT_ID)
    _require("R2_ACCESS_KEY_ID", R2_ACCESS_KEY_ID)
    _require("R2_SECRET_ACCESS_KEY", R2_SECRET_ACCESS_KEY)
    _require("R2_BUCKET", R2_BUCKET)
    endpoint = f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com"
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=R2_ACCESS_KEY_ID,
        aws_secret_access_key=R2_SECRET_ACCESS_KEY,
        config=Config(signature_version="s3v4", region_name="auto"),
    )


def content_type_for(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext in CONTENT_TYPE_OVERRIDES:
        return CONTENT_TYPE_OVERRIDES[ext]
    guess, _ = mimetypes.guess_type(path)
    return guess or "application/octet-stream"


def md5_of_file(path: str, chunk: int = 1024 * 1024) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            buf = f.read(chunk)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def head_etag(client, key: str) -> Optional[str]:
    """Return the ETag of an existing object, or None if it doesn't exist.
    R2 returns ETags as quoted hex strings — we strip the quotes."""
    try:
        resp = client.head_object(Bucket=R2_BUCKET, Key=key)
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        if code in ("404", "NoSuchKey", "NotFound"):
            return None
        raise
    return (resp.get("ETag") or "").strip('"')


def upload_file(
    client,
    local_path: str,
    key: str,
    cache_control: str = "public, max-age=86400",
    skip_if_unchanged: bool = True,
) -> str:
    """Upload local_path to <bucket>/<key>. Returns one of:
        'uploaded'  — actually wrote new bytes
        'skipped'   — remote ETag matched local MD5, no upload needed
    """
    if skip_if_unchanged:
        local_md5 = md5_of_file(local_path)
        remote_etag = head_etag(client, key)
        # Multipart uploads have ETags like "<md5>-<N>" so they won't match a
        # plain md5 — that's fine, we just re-upload in that case (rare here).
        if remote_etag and remote_etag == local_md5:
            return "skipped"

    extra = {
        "ContentType": content_type_for(local_path),
        "CacheControl": cache_control,
    }
    with open(local_path, "rb") as f:
        client.put_object(Bucket=R2_BUCKET, Key=key, Body=f, **extra)
    return "uploaded"


def public_url(key: str) -> str:
    base = _require("R2_PUBLIC_BASE", R2_PUBLIC_BASE)
    return f"{base}/{key.lstrip('/')}"
