"""SPEC-008 Phase C2: assemble the 8 pilot WSIs under <data-root>/wsi (D-9). Fails closed.

    uv run --offline python projects/cohorts/surgen-msi/c2-pilot/c2_wsi.py <command> \
        --data-root ~/pathology-ai-data/surgen-msi --c2-authorization <owner decision id> \
        --checkpoint <uni.pth> --legacy-root <legacy SurGen cache>

    acquire      download the 5 approved WSIs from EBI by HTTPS; resumable, range-validated
    clone-local  APFS-clone the 3 inventoried local WSIs from the legacy cache (read only)
    provenance   write runs/c2-pilot/acquisition/wsi-provenance.json for all 8 (write once)

`acquire` and `clone-local` run preflight[download] first and stop unless it prints GO. Every
command is idempotent: an interrupted run is resumed by re-running the same command.

Invariants:
- Only the 5 allowlisted URLs are requested; any status other than 200 (HEAD, full GET) or 206
  with the exact expected Content-Range (resumed GET) is refused, so a redirect is never stored.
- A resume sends If-Range with the strong ETag (else Last-Modified) recorded at first contact;
  bytes from a response that fails validation are truncated away before anything else happens.
- After completion every resume seam is re-fetched by range and compared byte for byte.
- A complete file is pinned by: SHA-256 journal -> chmod a-w -> link(2) into wsi/ (never
  clobbers) -> SHA256SUMS replaced by rename -> unlink the temporary -> chflags uchg.
  Each step is re-checked on re-run, so an interruption at any point is recoverable.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import platform
import re
import shutil
import socket
import stat
import subprocess
import sys
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, os.fspath(Path(__file__).resolve().parent))

from c2_preflight import (  # noqa: E402  (sibling module, resolved from this directory)
    BINDING_FILE,
    INVENTORY_ID,
    LOCAL,
    PENDING,
    PILOT,
    RESERVE_BYTES,
    SELECTION_ID,
    STAGING_ARTIFACT,
    SUMS,
    SUMS_NEXT,
    parse_sums,
    sha256,
)

BASE = "https://ftp.ebi.ac.uk/biostudies/fire/S-BIAD/285/S-BIAD1285/Files"
# Source: legacy data/metadata/SurGen/MMR_MSI/download_{train,validate}.txt (verified locally).
ALLOW = {s: f"{BASE}/{s.split('_', 1)[0]}_WSIs/{s}.czi" for s in PENDING}
CZI_MAGIC = b"ZISRAWFILE"  # first segment id of every CZI file
MAX_ATTEMPTS = 5
RETRY_DELAY_S = 30
SEAM_BYTES = 64 * 1024  # bytes re-fetched on each side of a resume offset
SHA256_SOURCE = (
    "computed locally over the complete file; trust on first download "
    "(no publisher checksum in the S-BIAD1285 study metadata)"
)


class Refused(RuntimeError):
    """A condition the operator must look at. Nothing is deleted or overwritten."""


class Transient(RuntimeError):
    """A network failure that a later attempt may resolve."""


def now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass(frozen=True)
class Layout:
    data: Path

    @property
    def stage(self) -> Path:
        return self.data / "staging" / "ebi"

    @property
    def wsi(self) -> Path:
        return self.data / "wsi"

    @property
    def prov(self) -> Path:
        return self.data / "runs" / "c2-pilot" / "acquisition"

    def artifact(self, name: str, suffix: str) -> Path:
        path = self.stage / f"{name}.{suffix}"
        if not STAGING_ARTIFACT.match(path.name):  # keeps preflight G3 and this tool in step
            raise AssertionError(f"{path.name} is not an allowlisted staging artifact")
        return path

    def pins(self) -> dict[str, str]:
        path = self.wsi / SUMS
        return parse_sums(path.read_text()) if path.exists() else {}

    def make(self) -> None:
        for d in (self.stage, self.prov):
            d.mkdir(mode=0o700, parents=True, exist_ok=True)
        if not self.wsi.exists():
            self.wsi.mkdir(mode=0o700, parents=True)


@dataclass
class Settings:
    curl: tuple[str, ...] = ("curl",)  # tests add --cacert/--connect-to for a local mock
    retry_delay_s: float = RETRY_DELAY_S
    reserve_bytes: int = RESERVE_BYTES
    max_attempts: int = MAX_ATTEMPTS
    seam_bytes: int = SEAM_BYTES
    context: dict[str, str] = field(default_factory=dict)  # recorded in every provenance line
    log: Callable[[str], None] = lambda m: print(f"c2_wsi: {m}", flush=True)


# --------------------------------------------------------------------------- files


def write_json(path: Path, doc: dict[str, Any]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w") as f:
        json.dump(doc, f, indent=1, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def append_jsonl(path: Path, doc: dict[str, Any]) -> None:
    with path.open("a") as f:
        f.write(json.dumps(doc, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


def jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []


def size(path: Path) -> int:
    return path.stat().st_size if path.exists() else 0


def is_sealed(path: Path) -> bool:
    st = path.lstat()
    flags = getattr(st, "st_flags", stat.UF_IMMUTABLE)
    return not st.st_mode & 0o222 and bool(flags & stat.UF_IMMUTABLE)


def seal(path: Path) -> None:
    """a-w, plus the user-immutable flag where the OS has one (no rename, unlink or write)."""
    os.chmod(path, 0o444)
    if hasattr(os, "chflags"):
        os.chflags(path, os.lstat(path).st_flags | stat.UF_IMMUTABLE)


@contextlib.contextmanager
def writable(directory: Path) -> Iterator[None]:
    os.chmod(directory, 0o700)
    try:
        yield
    finally:
        os.chmod(directory, 0o500)


@contextlib.contextmanager
def locked(layout: Layout) -> Iterator[None]:
    lock = layout.prov / "c2_wsi.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        raise Refused(
            f"{lock} exists ({lock.read_text().strip()}): another run is active, or one was "
            "killed; check `ps` and remove the lock by hand"
        ) from None
    os.write(fd, f"pid {os.getpid()} since {now()}".encode())
    os.close(fd)
    try:
        yield
    finally:
        lock.unlink()


def disk_ok(layout: Layout, planned: int, s: Settings) -> None:
    free = shutil.disk_usage(layout.data).free
    if free - planned < s.reserve_bytes:
        raise Refused(f"{planned} more bytes would breach the {s.reserve_bytes} B reserve")


# --------------------------------------------------------------------------- HTTP (curl)


@dataclass(frozen=True)
class Response:
    status: int
    headers: dict[str, str]

    def get(self, key: str) -> str:
        return self.headers.get(key.lower(), "")


def parse_headers(text: str) -> Response | None:
    """The last response block of a curl --dump-header file (lower-cased header names)."""
    blocks = [b for b in re.split(r"\r?\n\r?\n", text) if b.strip().startswith("HTTP/")]
    if not blocks:
        return None
    lines = blocks[-1].strip().splitlines()
    status = int(lines[0].split()[1])
    headers: dict[str, str] = {}
    for line in lines[1:]:
        k, sep, v = line.partition(":")
        if sep:
            headers[k.strip().lower()] = v.strip()
    return Response(status, headers)


def curl_base(s: Settings) -> list[str]:
    return [
        *s.curl,
        "--silent",
        "--show-error",
        "--proto",
        "=https",
        "--proto-redir",
        "=https",
        "--max-redirs",
        "0",
        "--tlsv1.2",
        "--connect-timeout",
        "60",
    ]


def head(url: str, hdr: Path, s: Settings) -> Response:
    p = subprocess.run(
        [*curl_base(s), "--head", "--dump-header", str(hdr), "--output", os.devnull, url],
        capture_output=True,
        text=True,
    )
    r = parse_headers(hdr.read_text()) if hdr.exists() else None
    if p.returncode != 0 or r is None:
        raise Transient(f"HEAD failed (curl {p.returncode}: {p.stderr.strip()})")
    return r


def get(
    url: str, out: Path, hdr: Path, s: Settings, *, start: int, validator: str | None
) -> tuple[int, str, Response | None]:
    """GET into `out`, appending from `start` (curl -C refuses a non-206 answer)."""
    args = [*curl_base(s), "--dump-header", str(hdr), "--output", str(out)]
    args += ["--speed-limit", "10240", "--speed-time", "300"]
    if start:
        args += ["--continue-at", str(start)]
    if validator:
        args += ["--header", f"If-Range: {validator}"]
    p = subprocess.run([*args, url], capture_output=True, text=True)
    return p.returncode, p.stderr.strip(), parse_headers(hdr.read_text()) if hdr.exists() else None


def get_range(url: str, out: Path, hdr: Path, s: Settings, first: int, last: int) -> Response:
    if out.exists():
        out.unlink()  # a re-fetched comparison buffer, never data
    args = [*curl_base(s), "--range", f"{first}-{last}", "--dump-header", str(hdr)]
    p = subprocess.run([*args, "--output", str(out), url], capture_output=True, text=True)
    r = parse_headers(hdr.read_text()) if hdr.exists() else None
    if p.returncode != 0 or r is None:
        raise Transient(f"range {first}-{last} failed (curl {p.returncode}: {p.stderr.strip()})")
    return r


def remote_of(r: Response) -> dict[str, Any]:
    if r.status != 200:
        raise Refused(f"HEAD answered {r.status}, not 200 (a redirect is never followed)")
    length = r.get("content-length")
    if not length.isdigit() or int(length) < len(CZI_MAGIC):
        raise Refused(f"no usable Content-Length ({length!r})")
    return {
        "content_length": int(length),
        "etag": r.get("etag"),
        "etag_is_digest": False,
        "last_modified": r.get("last-modified"),
        "accept_ranges": r.get("accept-ranges"),
    }


def validator_of(meta: dict[str, Any]) -> str | None:
    etag = str(meta["etag"])
    if etag and not etag.startswith("W/"):
        return etag
    return str(meta["last_modified"]) or None


def same_remote(meta: dict[str, Any], r: Response) -> bool:
    """Same validators; for a 200 answer also the same length (a 206 carries a range length)."""
    if r.status == 200 and str(meta["content_length"]) != r.get("content-length"):
        return False
    return (meta["etag"], meta["last_modified"]) == (r.get("etag"), r.get("last-modified"))


def get_problem(r: Response | None, have: int, meta: dict[str, Any]) -> str | None:
    """Why the bytes of this GET must not be kept, or None when they extend the .part."""
    if r is None:
        return "no response headers"
    if (r.get("etag"), r.get("last-modified")) != (meta["etag"], meta["last_modified"]):
        return "remote changed (ETag or Last-Modified)"
    length = meta["content_length"]
    if have == 0:
        if r.status != 200 or r.get("content-length") != str(length):
            return f"status {r.status}, Content-Length {r.get('content-length')!r}"
        return None
    expected = f"bytes {have}-{length - 1}/{length}"
    if r.status != 206 or r.get("content-range") != expected:
        return f"status {r.status}, Content-Range {r.get('content-range')!r} != {expected!r}"
    return None


# --------------------------------------------------------------------------- acquisition


def download(layout: Layout, slide: str, url: str, s: Settings) -> None:
    """Bring staging/<name>.part to the full Content-Length, resuming across attempts/runs."""
    name = f"{slide}.czi"
    part, meta_path = layout.artifact(name, "part"), layout.artifact(name, "transfer.json")
    meta: dict[str, Any] | None = json.loads(meta_path.read_text()) if meta_path.exists() else None
    for _ in range(s.max_attempts):
        n = len(meta["attempts"]) + 1 if meta else 1
        record: dict[str, Any] = {"attempt": n, "started_utc": now()}
        try:
            r = head(url, layout.artifact(name, f"a{n:03d}.head"), s)
            if meta is None:
                meta = {
                    "slide_id": slide,
                    "file_name": name,
                    "url": url,
                    **remote_of(r),
                    "first_seen_utc": record["started_utc"],
                    "attempts": [],
                    "resume_offsets": [],
                }
                write_json(meta_path, meta)
            elif not same_remote(meta, r) or r.status != 200:
                raise Refused(
                    f"{name}: the remote object changed since {meta['first_seen_utc']} "
                    f"(HEAD {r.status}); move {part.name} aside by hand before a new transfer"
                )
            length, have = int(meta["content_length"]), size(part)
            if have > length:
                raise Refused(f"{name}: .part ({have} B) is larger than Content-Length {length}")
            if have == length:
                return
            validator = validator_of(meta)
            if have and (meta["accept_ranges"] != "bytes" or validator is None):
                raise Refused(f"{name}: cannot resume safely (no byte ranges or no validator)")
            disk_ok(layout, length - have, s)
            hdr = layout.artifact(name, f"a{n:03d}.get")
            rc, err, resp = get(url, part, hdr, s, start=have, validator=validator)
            problem = get_problem(resp, have, meta)
            got = size(part) - have
            record.update(
                curl_exit=rc, status=resp and resp.status, bytes=max(got, 0), error=err or None
            )
            if problem is not None:
                if part.exists():
                    os.truncate(part, have)  # drop every byte of the rejected response
                record.update(discarded=problem, bytes=0)
                if resp is not None:  # a wrong answer, not a lost connection: stop
                    raise Refused(f"{name}: {problem}; the valid {have} B prefix is kept")
                raise Transient(problem)
            if have and got > 0:
                meta["resume_offsets"].append(have)
            if rc != 0:
                raise Transient(f"curl {rc}: {err}")
            if size(part) != length:
                raise Transient(f"short transfer ({size(part)}/{length} B)")
            return
        except Transient as exc:
            record.setdefault("error", str(exc))
            s.log(f"{name}: attempt {n} interrupted: {exc}")
            time.sleep(s.retry_delay_s)
        finally:
            if meta is not None:
                meta["attempts"].append({**record, "ended_utc": now()})
                write_json(meta_path, meta)
    raise Refused(f"{name}: not complete after {s.max_attempts} attempts; re-run to resume")


def verify_content(layout: Layout, slide: str, url: str, s: Settings) -> None:
    """CZI magic, then re-fetch every resume seam by range and compare byte for byte."""
    name = f"{slide}.czi"
    part, meta_path = layout.artifact(name, "part"), layout.artifact(name, "transfer.json")
    meta = json.loads(meta_path.read_text())
    with part.open("rb") as f:
        if f.read(len(CZI_MAGIC)) != CZI_MAGIC:
            raise Refused(f"{name}: not a CZI file (magic bytes differ)")
    length, checked = int(meta["content_length"]), []
    for k, offset in enumerate(sorted(set(meta["resume_offsets"]))):
        first, last = max(0, offset - s.seam_bytes), min(length - 1, offset + s.seam_bytes - 1)
        buf = layout.artifact(name, f"seam{k:02d}.bytes")
        r = get_range(url, buf, layout.artifact(name, f"seam{k:02d}.headers"), s, first, last)
        expected = f"bytes {first}-{last}/{length}"
        if r.status != 206 or r.get("content-range") != expected or not same_remote(meta, r):
            raise Refused(f"{name}: seam {offset} re-fetch answered {r.status} {r.headers}")
        with part.open("rb") as f:
            f.seek(first)
            if f.read(last - first + 1) != buf.read_bytes():
                raise Refused(f"{name}: bytes around resume offset {offset} differ from EBI")
        checked.append(offset)
    meta["seams_verified"] = checked
    write_json(meta_path, meta)


def pin(layout: Layout, src: Path, name: str, sha: str) -> None:
    """Link a verified temporary into wsi/ and pin it. Safe to repeat after any interruption."""
    dst, pins = layout.wsi / name, layout.pins()
    if name in pins:
        if pins[name] != sha:
            raise Refused(f"{name} is pinned with another SHA-256")
    else:
        os.chmod(src, 0o444)
        with writable(layout.wsi):
            if dst.exists() and not os.path.samefile(src, dst):
                raise Refused(f"wsi/{name} exists and is not the verified temporary")
            if not dst.exists():
                os.link(src, dst)  # link(2) fails rather than replace an existing name
            text = (layout.wsi / SUMS).read_text() if (layout.wsi / SUMS).exists() else ""
            nxt = layout.stage / SUMS_NEXT
            if nxt.exists():  # left by an interruption before the rename; rebuilt from pins
                nxt.unlink()
            with nxt.open("w") as f:
                f.write(text + f"{sha}  {name}\n")
                f.flush()
                os.fsync(f.fileno())
            parse_sums(nxt.read_text())  # refuses a duplicate or malformed line
            os.chmod(nxt, 0o444)
            os.replace(nxt, layout.wsi / SUMS)  # atomic; needs only the directory write bit
    if src.exists():
        if not os.path.samefile(src, dst):
            raise Refused(f"{src.name} is not wsi/{name}; leaving both for inspection")
        src.unlink()  # the bytes stay in wsi/ (same inode)
    if not is_sealed(dst):
        seal(dst)
    os.chmod(layout.wsi / SUMS, 0o444)
    os.chmod(layout.wsi, 0o500)


def journal(layout: Layout, name: str, src: Path) -> dict[str, Any]:
    path = layout.artifact(name, "verified.json")
    if path.exists():
        doc: dict[str, Any] = json.loads(path.read_text())
    else:
        doc = {"sha256": sha256(src), "size_bytes": size(src), "hashed_utc": now()}
        write_json(path, doc)
    return doc


def archive(layout: Layout, slide: str) -> None:
    """Move the slide's staging artifacts into the provenance tree (nothing is deleted)."""
    dest = layout.prov / slide
    dest.mkdir(mode=0o700, exist_ok=True)
    for p in sorted(layout.stage.glob(f"{slide}.czi.*")):
        target = dest / p.name
        if target.exists():
            raise Refused(f"{target} already exists")
        os.rename(p, target)


Record = Callable[[dict[str, Any]], dict[str, Any]]


def complete(
    layout: Layout, slide: str, src_suffix: str, expected: str | None, record: Record, ledger: str
) -> str:
    """Journal -> pin -> provenance -> archive; every step is skipped when already done."""
    name = f"{slide}.czi"
    src, dst = layout.artifact(name, src_suffix), layout.wsi / name
    pinned = layout.pins().get(name)
    if pinned is None:
        if not src.exists():
            raise Refused(f"wsi/{name} without a pin and without its temporary")
        j = journal(layout, name, src)
        if sha256(src) != j["sha256"] or size(src) != j["size_bytes"]:
            raise Refused(f"{src.name} changed after it was hashed")
        if expected is not None and j["sha256"] != expected:
            raise Refused(f"{name}: SHA-256 {j['sha256'][:12]}… != inventory {expected[:12]}…")
        pin(layout, src, name, j["sha256"])
        pinned = j["sha256"]
    elif src.exists():
        pin(layout, src, name, pinned)  # finishes an interrupted unlink and seal
    if not is_sealed(dst):
        seal(dst)
    if not any(x["slide_id"] == slide for x in jsonl(layout.prov / ledger)):
        append_jsonl(layout.prov / ledger, record({"sha256": pinned, "size_bytes": size(dst)}))
    archive(layout, slide)
    return pinned


def acquire(layout: Layout, s: Settings) -> None:
    layout.make()
    for slide in PENDING:
        name, url = f"{slide}.czi", ALLOW[slide]
        if not url.startswith(BASE + "/") or not url.endswith("/" + name):
            raise Refused(f"{url} does not name {name}")
        pinned = name in layout.pins()
        if not pinned and not layout.artifact(name, "verified.json").exists():
            download(layout, slide, url, s)
            verify_content(layout, slide, url, s)
        meta_path = layout.artifact(name, "transfer.json")

        def record(extra: dict[str, Any], meta_path: Path = meta_path) -> dict[str, Any]:
            meta = json.loads(meta_path.read_text())
            return {
                **{k: v for k, v in meta.items() if k != "attempts"},
                **extra,
                "attempts": len(meta["attempts"]),
                "sha256_source": SHA256_SOURCE,
                "completed_utc": now(),
                "decision": "D-9",
                "selection": SELECTION_ID,
                **s.context,
            }

        sha = complete(layout, slide, "part", None, record, "transfers.jsonl")
        s.log(f"{name} verified and sealed (sha256 {sha})")
    s.log(f"{len(PENDING)}/{len(PENDING)} approved downloads pinned and sealed")


def clone_local(layout: Layout, legacy: Path, s: Settings) -> None:
    layout.make()
    for slide, (rel, expected) in sorted(LOCAL.items()):
        name = f"{slide}.czi"
        src = layout.artifact(name, "clone")
        if name not in layout.pins() and not src.exists():
            disk_ok(layout, 0, s)
            # -c: clonefile(2) only (no data copy; fails across volumes); the source is read.
            subprocess.run(["cp", "-c", str(legacy / rel), str(src)], check=True)

        def record(
            extra: dict[str, Any], slide: str = slide, name: str = name, rel: str = rel
        ) -> dict[str, Any]:
            return {
                "slide_id": slide,
                "file_name": name,
                "source": "legacy-cache (read only)",
                "legacy_path": rel,
                "method": "APFS clone (cp -c, clonefile(2))",
                "sha256_source": f"equal to inventory {INVENTORY_ID} (content-verified in C1)",
                "upstream_url_not_fetched": f"{BASE}/{slide.split('_', 1)[0]}_WSIs/{name}",
                "completed_utc": now(),
                "decision": "D-9",
                "selection": SELECTION_ID,
                **extra,
                **s.context,
            }

        sha = complete(layout, slide, "clone", expected, record, "local-clones.jsonl")
        s.log(f"{name} cloned, verified against the inventory and sealed (sha256 {sha})")


def provenance(layout: Layout) -> Path:
    """One record per pilot WSI, cross-checked with SHA256SUMS and the files (write once)."""
    pins = layout.pins()
    records = {
        r["slide_id"]: {"kind": kind, **r}
        for kind, file in (("ebi-https", "transfers.jsonl"), ("local-clone", "local-clones.jsonl"))
        for r in jsonl(layout.prov / file)
    }
    problems = []
    if set(pins) != {f"{x}.czi" for x in PILOT} or set(records) != set(PILOT):
        problems.append(f"{len(pins)} pins and {len(records)} records, not 8 and 8")
    for slide, r in records.items():
        name = f"{slide}.czi"
        if pins.get(name) != r["sha256"] or size(layout.wsi / name) != r["size_bytes"]:
            problems.append(f"{slide}: record disagrees with {SUMS} or the file")
        if r["kind"] == "ebi-https" and r["size_bytes"] != r["content_length"]:
            problems.append(f"{slide}: size != Content-Length")
    if problems:
        raise Refused("; ".join(problems))
    doc = {
        "schema": "pathology-ai.c2-wsi-provenance/1",
        "selection": SELECTION_ID,
        "wsi": [records[x] for x in sorted(records)],
    }
    out = layout.prov / "wsi-provenance.json"
    if out.exists():
        if json.loads(out.read_text()) != doc:
            raise Refused(f"{out} exists with other content; it is never overwritten")
        return out
    write_json(out, doc)
    os.chmod(out, 0o444)
    return out


# --------------------------------------------------------------------------- CLI


def preflight(a: argparse.Namespace) -> None:
    cmd = [
        sys.executable,
        os.fspath(Path(__file__).resolve().parent / "c2_preflight.py"),
        "--stage",
        "download",
        "--repo",
        os.fspath(a.repo),
        "--data-root",
        os.fspath(a.data_root),
        "--checkpoint",
        os.fspath(a.checkpoint),
        "--legacy-root",
        os.fspath(a.legacy_root),
        "--c2-authorization",
        a.c2_authorization,
    ]
    p = subprocess.run(cmd, capture_output=True, text=True)
    sys.stdout.write(p.stdout)
    if p.returncode != 0 or "preflight[download]: GO" not in p.stdout:
        raise Refused("preflight[download] is NO-GO")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    p.add_argument("command", choices=("acquire", "clone-local", "provenance"))
    p.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[4])
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--legacy-root", type=Path)
    p.add_argument("--c2-authorization")
    a = p.parse_args(argv)
    os.umask(0o077)
    a.repo, a.data_root = a.repo.resolve(), a.data_root.expanduser()
    layout = Layout(a.data_root)
    try:
        if a.command == "provenance":
            layout.make()
            with locked(layout):
                print(f"c2_wsi: wrote {provenance(layout)}")
            return 0
        if not (a.checkpoint and a.legacy_root and a.c2_authorization):
            raise Refused("--checkpoint, --legacy-root and --c2-authorization are required")
        a.checkpoint, a.legacy_root = a.checkpoint.expanduser(), a.legacy_root.expanduser()
        preflight(a)
        version = subprocess.run(["curl", "--version"], capture_output=True, text=True, check=True)
        s = Settings(
            context={
                "c2_authorization": a.c2_authorization,
                "c2_binding_sha256": sha256(a.repo / BINDING_FILE),
                "client": version.stdout.splitlines()[0],
                "host": socket.gethostname(),
                "platform": platform.platform(),
            }
        )
        layout.make()
        with locked(layout):
            if a.command == "acquire":
                acquire(layout, s)
            else:
                clone_local(layout, a.legacy_root, s)
        return 0
    except (Refused, subprocess.CalledProcessError) as exc:
        print(f"c2_wsi: REFUSED: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    sys.exit(main())
