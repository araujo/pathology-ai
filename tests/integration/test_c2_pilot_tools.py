"""SPEC-008 Phase C2 tools (PROMPT-008D10): acquisition, resume, pinning, binding, re-selection.

Every download runs real curl over real TLS against a local mock of ftp.ebi.ac.uk (self-signed
certificate, `--connect-to`). No request leaves this host; no real WSI is read or written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import ssl
import stat
import subprocess
import sys
import threading
from collections.abc import Iterator
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "projects/cohorts/surgen-msi/c2-pilot"
sys.path.insert(0, str(TOOLS))

import c2_preflight as pf  # noqa: E402
import c2_wsi as w  # noqa: E402

pytestmark = pytest.mark.skipif(
    sys.platform != "darwin" or not shutil.which("curl") or not shutil.which("openssl"),
    reason="C2 tools target macOS (APFS clones, chflags) and need curl and openssl",
)
HOST = "ftp.ebi.ac.uk"
SIZE = 300_000


# --------------------------------------------------------------------------- mock EBI


@dataclass
class Blob:
    data: bytes
    etag: str
    modified: str = "Wed, 24 Jul 2024 10:00:00 GMT"


@dataclass
class MockEBI:
    blobs: dict[str, Blob]
    faults: dict[str, list[str]] = field(default_factory=dict)  # per path, one per GET
    requests: list[dict[str, str]] = field(default_factory=list)


def _handler(mock: MockEBI) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args: Any) -> None:
            pass

        def _head(self, status: int, blob: Blob, length: int, extra: dict[str, str]) -> None:
            self.send_response(status)
            for k, v in {
                "Content-Length": str(length),
                "ETag": blob.etag,
                "Last-Modified": blob.modified,
                "Accept-Ranges": "bytes",
                "Connection": "close",
                **extra,
            }.items():
                self.send_header(k, v)
            self.end_headers()

        def do_HEAD(self) -> None:  # noqa: N802
            self._serve(body=False)

        def do_GET(self) -> None:  # noqa: N802
            self._serve(body=True)

        def _serve(self, body: bool) -> None:
            path = self.path
            mock.requests.append(
                {
                    "method": self.command,
                    "path": path,
                    "range": self.headers.get("Range", ""),
                    "if_range": self.headers.get("If-Range", ""),
                }
            )
            blob = mock.blobs.get(path)
            fault = mock.faults.get(path, []).pop(0) if body and mock.faults.get(path) else ""
            if blob is None or (not body and mock.faults.get(path + "#head")):
                self.send_response(301 if blob else 404)
                self.send_header("Location", "https://elsewhere.example/x")
                self.send_header("Content-Length", "162")
                self.send_header("Connection", "close")
                self.end_headers()
                return
            data, n = blob.data, len(blob.data)
            first, last = 0, n - 1
            ranged = self.headers.get("Range", "").startswith("bytes=")
            if ranged and self.headers.get("If-Range", blob.etag) not in (blob.etag, blob.modified):
                ranged = False  # RFC 9110: a stale validator gets the full representation
            if ranged and fault != "ignore-range":
                a, _, b = self.headers["Range"][6:].partition("-")
                first, last = int(a), int(b) if b else n - 1
                if fault == "shift-range":
                    first += 1
                chunk = bytearray(data[first : last + 1])
                if fault == "corrupt" and chunk:
                    chunk[0] ^= 0xFF
                cr = {"Content-Range": f"bytes {first}-{last}/{n}"}
                self._head(206, blob, len(chunk), cr)
                payload = bytes(chunk)
            else:
                self._head(200, blob, n, {})
                payload = data
            if not body:
                return
            if fault.startswith("drop:"):
                self.wfile.write(payload[: int(fault[5:])])
                self.wfile.flush()
                self.connection.shutdown(2)  # abrupt end: curl reports a partial transfer
                return
            self.wfile.write(payload)

    return Handler


@pytest.fixture(scope="module")
def cert(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    d = tmp_path_factory.mktemp("tls")
    key, crt = d / "key.pem", d / "cert.pem"
    subprocess.run(
        ["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "2",
         "-keyout", str(key), "-out", str(crt), "-subj", f"/CN={HOST}",
         "-addext", f"subjectAltName=DNS:{HOST}"],
        check=True, capture_output=True,
    )  # fmt: skip
    return crt, key


def _content(slide: str) -> bytes:
    seed = hashlib.sha256(slide.encode()).digest()
    body = (seed * (SIZE // len(seed) + 1))[: SIZE - len(w.CZI_MAGIC)]
    return w.CZI_MAGIC + bytes((b + i) % 256 for i, b in enumerate(body))


def _path(slide: str) -> str:
    return w.ALLOW[slide].removeprefix(f"https://{HOST}")


@pytest.fixture
def ebi(cert: tuple[Path, Path]) -> Iterator[tuple[MockEBI, w.Settings, list[str]]]:
    mock = MockEBI({_path(s): Blob(_content(s), f'"{s}-v1"') for s in pf.PENDING})
    server = ThreadingHTTPServer(("127.0.0.1", 0), _handler(mock))
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(*cert)
    server.socket = ctx.wrap_socket(server.socket, server_side=True)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    port = server.server_address[1]
    logs: list[str] = []
    settings = w.Settings(
        curl=("curl", "--cacert", str(cert[0]), "--connect-to", f"{HOST}:443:127.0.0.1:{port}"),
        retry_delay_s=0,
        reserve_bytes=0,
        context={"c2_authorization": "D-TEST", "c2_binding_sha256": "0" * 64},
        log=logs.append,
    )
    yield mock, settings, logs
    server.shutdown()


@pytest.fixture
def layout(tmp_path: Path) -> Iterator[w.Layout]:
    lay = w.Layout(tmp_path / "data")
    yield lay
    for p in sorted(tmp_path.rglob("*"), key=lambda p: -len(p.parts)):  # undo seals
        if hasattr(os, "chflags"):
            os.chflags(p, 0, follow_symlinks=False)
        if not p.is_symlink():
            os.chmod(p, 0o700)
    os.chmod(tmp_path, 0o700)


def _gate(gate: Any, data: Path, stage: str = "download", **kw: Any) -> tuple[str, str]:
    a = argparse.Namespace(repo=ROOT, data_root=data, stage=stage, legacy_root=None, **kw)
    result: tuple[str, str] = gate(a)
    return result


def _expected_pins() -> dict[str, str]:
    return {f"{s}.czi": hashlib.sha256(_content(s)).hexdigest() for s in pf.PENDING}


# --------------------------------------------------------------------------- acquisition


def test_five_sequential_downloads_are_pinned_sealed_and_recorded(
    ebi: tuple[MockEBI, w.Settings, list[str]], layout: w.Layout
) -> None:
    """Regression: SHA256SUMS used to be a-w after the first pin, so download 2 failed."""
    mock, s, _ = ebi
    w.acquire(layout, s)
    assert layout.pins() == _expected_pins()
    for name in layout.pins():
        assert w.is_sealed(layout.wsi / name)
    assert not os.stat(layout.wsi / pf.SUMS).st_mode & 0o222
    assert not os.stat(layout.wsi).st_mode & 0o222
    assert list(layout.stage.iterdir()) == []  # every artifact archived
    rows = w.jsonl(layout.prov / "transfers.jsonl")
    assert [r["slide_id"] for r in rows] == list(pf.PENDING)
    for r in rows:
        assert r["size_bytes"] == r["content_length"] == SIZE
        assert r["etag_is_digest"] is False and "trust on first download" in r["sha256_source"]
        assert r["resume_offsets"] == [] and r["c2_authorization"] == "D-TEST"
        assert (layout.prov / r["slide_id"] / f"{r['file_name']}.a001.head").is_file()
    assert {q["range"] for q in mock.requests} == {""}  # no resume was needed
    # Idempotent: a second run sends no request and changes nothing.
    before = len(mock.requests)
    w.acquire(layout, s)
    assert len(mock.requests) == before and layout.pins() == _expected_pins()
    assert len(w.jsonl(layout.prov / "transfers.jsonl")) == 5
    assert _gate(pf.g_no_test, layout.data)[0] == "PASS"
    assert _gate(pf.g_immutable, layout.data)[0] == "PASS"


def test_interrupted_transfer_resumes_with_if_range_and_verified_seam(
    ebi: tuple[MockEBI, w.Settings, list[str]], layout: w.Layout
) -> None:
    mock, s, logs = ebi
    slide = pf.PENDING[2]
    mock.faults[_path(slide)] = ["drop:100000"]
    w.acquire(layout, s)
    assert layout.pins() == _expected_pins()
    gets = [q for q in mock.requests if q["path"] == _path(slide) and q["method"] == "GET"]
    assert gets[1]["range"] == "bytes=100000-" and gets[1]["if_range"] == f'"{slide}-v1"'
    assert gets[2]["range"] == f"bytes={100000 - w.SEAM_BYTES}-{100000 + w.SEAM_BYTES - 1}"
    row = next(r for r in w.jsonl(layout.prov / "transfers.jsonl") if r["slide_id"] == slide)
    assert row["resume_offsets"] == [100000] and row["seams_verified"] == [100000]
    assert row["attempts"] == 2 and any("interrupted" in m for m in logs)


def test_resume_across_runs_passes_preflight_staging_allowlist(
    ebi: tuple[MockEBI, w.Settings, list[str]], layout: w.Layout
) -> None:
    """Regression: the acquisition's own artifacts used to fail G3, blocking any resume."""
    mock, s, _ = ebi
    slide = pf.PENDING[0]
    mock.faults[_path(slide)] = ["drop:50000"]
    s.max_attempts = 1
    with pytest.raises(w.Refused, match="not complete after 1 attempts"):
        w.acquire(layout, s)
    staged = sorted(p.name for p in layout.stage.iterdir())
    assert f"{slide}.czi.part" in staged and f"{slide}.czi.a001.get" in staged
    assert _gate(pf.g_no_test, layout.data) == (
        "PASS",
        "partitions ['train', 'validation']; no file outside the 8 pilot slides",
    )
    assert _gate(pf.g_no_test, layout.data, stage="extract")[0] == "FAIL"  # not finished
    s.max_attempts = w.MAX_ATTEMPTS
    w.acquire(layout, s)  # a new run resumes from the kept .part
    assert layout.pins() == _expected_pins()
    meta = json.loads((layout.prov / slide / f"{slide}.czi.transfer.json").read_text())
    assert meta["resume_offsets"] == [50000] and len(meta["attempts"]) == 2


@pytest.mark.parametrize(
    ("fault", "message"),
    [
        ("ignore-range", "status 200"),  # server or proxy ignores Range
        ("shift-range", "Content-Range 'bytes 40001-"),  # wrong offset must never be appended
    ],
)
def test_invalid_resume_answer_is_refused_and_only_the_valid_prefix_kept(
    ebi: tuple[MockEBI, w.Settings, list[str]], layout: w.Layout, fault: str, message: str
) -> None:
    mock, s, _ = ebi
    slide = pf.PENDING[1]
    mock.faults[_path(slide)] = ["drop:40000", fault]
    with pytest.raises(w.Refused, match=message):
        w.acquire(layout, s)
    part = layout.stage / f"{slide}.czi.part"
    assert part.read_bytes() == _content(slide)[:40000]
    assert f"{slide}.czi" not in layout.pins()


def test_remote_change_between_attempts_is_refused(
    ebi: tuple[MockEBI, w.Settings, list[str]], layout: w.Layout
) -> None:
    mock, s, _ = ebi
    slide = pf.PENDING[3]
    mock.faults[_path(slide)] = ["drop:30000"]
    s.max_attempts = 1
    with pytest.raises(w.Refused):
        w.acquire(layout, s)
    mock.blobs[_path(slide)].etag = f'"{slide}-v2"'  # republished upstream
    s.max_attempts = w.MAX_ATTEMPTS
    with pytest.raises(w.Refused, match="remote object changed"):
        w.acquire(layout, s)
    assert (layout.stage / f"{slide}.czi.part").read_bytes() == _content(slide)[:30000]


def test_corrupted_bytes_at_a_resume_seam_are_detected(
    ebi: tuple[MockEBI, w.Settings, list[str]], layout: w.Layout
) -> None:
    """A 206 with valid headers but wrong bytes passes size checks; the seam re-fetch fails."""
    mock, s, _ = ebi
    slide = pf.PENDING[4]
    mock.faults[_path(slide)] = ["drop:120000", "corrupt"]
    with pytest.raises(w.Refused, match="bytes around resume offset 120000 differ"):
        w.acquire(layout, s)
    assert f"{slide}.czi" not in layout.pins() and not (layout.wsi / f"{slide}.czi").exists()


def test_redirect_is_refused_and_never_stored(
    ebi: tuple[MockEBI, w.Settings, list[str]], layout: w.Layout
) -> None:
    """Regression: --max-redirs 0 without -L returns the 3xx body with exit 0."""
    mock, s, _ = ebi
    mock.faults[_path(pf.PENDING[0]) + "#head"] = ["redirect"]
    with pytest.raises(w.Refused, match="HEAD answered 301"):
        w.acquire(layout, s)
    assert layout.pins() == {}
    assert not (layout.stage / f"{pf.PENDING[0]}.czi.part").exists()


def test_not_a_czi_is_refused(ebi: tuple[MockEBI, w.Settings, list[str]], layout: w.Layout) -> None:
    mock, s, _ = ebi
    mock.blobs[_path(pf.PENDING[0])].data = b"<html>" + b"x" * 1000
    with pytest.raises(w.Refused, match="not a CZI file"):
        w.acquire(layout, s)
    assert layout.pins() == {}


def test_crash_between_link_and_pin_recovers_without_duplicate_pins(
    ebi: tuple[MockEBI, w.Settings, list[str]],
    layout: w.Layout,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, s, _ = ebi
    real = os.replace
    calls = {"n": 0}

    def crash_on_second_pin(src: Any, dst: Any) -> None:
        if str(dst).endswith(pf.SUMS):
            calls["n"] += 1
            if calls["n"] == 2:
                raise KeyboardInterrupt  # power loss after link(2), before the pin
        real(src, dst)

    monkeypatch.setattr(os, "replace", crash_on_second_pin)
    with pytest.raises(KeyboardInterrupt):
        w.acquire(layout, s)
    second = f"{pf.PENDING[1]}.czi"
    assert (layout.wsi / second).exists() and second not in layout.pins()
    assert _gate(pf.g_no_test, layout.data)[0] == "PASS"
    assert pf.journal_exists(layout.data, second)  # G8 accepts this orphan only with a journal
    monkeypatch.setattr(os, "replace", real)
    os.chmod(layout.wsi, 0o500)
    w.acquire(layout, s)
    assert layout.pins() == _expected_pins()
    assert len((layout.wsi / pf.SUMS).read_text().splitlines()) == 5


def test_crash_before_archive_recovers_without_duplicate_provenance(
    ebi: tuple[MockEBI, w.Settings, list[str]],
    layout: w.Layout,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, s, _ = ebi
    real = w.archive

    def crash(*a: Any) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(w, "archive", crash)
    with pytest.raises(KeyboardInterrupt):
        w.acquire(layout, s)
    monkeypatch.setattr(w, "archive", real)
    w.acquire(layout, s)
    rows = w.jsonl(layout.prov / "transfers.jsonl")
    assert [r["slide_id"] for r in rows] == list(pf.PENDING)
    assert list(layout.stage.iterdir()) == []


def test_concurrent_run_is_refused(layout: w.Layout) -> None:
    layout.make()
    with w.locked(layout), pytest.raises(w.Refused, match="another run is active"):
        with w.locked(layout):
            pass
    assert not (layout.prov / "c2_wsi.lock").exists()


def test_every_staging_artifact_name_is_allowlisted_by_preflight(layout: w.Layout) -> None:
    name = f"{pf.PENDING[0]}.czi"
    for suffix in ("part", "clone", "transfer.json", "verified.json", "a001.head", "a012.get",
                   "seam00.bytes", "seam03.headers"):  # fmt: skip
        assert pf.STAGING_ARTIFACT.match(layout.artifact(name, suffix).name)
        assert pf.STAGING_ARTIFACT.match(layout.artifact(name, suffix).name + ".tmp")
    with pytest.raises(AssertionError):
        layout.artifact(name, "curl.log")
    assert not pf.STAGING_ARTIFACT.match("SR386_40X_HE_T044_02.czi.part")


# --------------------------------------------------------------------------- local clones


def _fake_local(tmp_path: Path) -> tuple[Path, dict[str, tuple[str, str]]]:
    legacy = tmp_path / "legacy"
    table = {}
    for slide, (rel, _) in pf.LOCAL.items():
        src = legacy / rel
        src.parent.mkdir(parents=True, exist_ok=True)
        src.write_bytes(_content(slide))
        os.chmod(src, 0o444)
        table[slide] = (rel, hashlib.sha256(_content(slide)).hexdigest())
    return legacy, table


def test_clone_local_verifies_against_the_inventory_and_leaves_sources_untouched(
    tmp_path: Path, layout: w.Layout, monkeypatch: pytest.MonkeyPatch
) -> None:
    legacy, table = _fake_local(tmp_path)
    monkeypatch.setattr(w, "LOCAL", table)
    before = {rel: os.stat(legacy / rel) for rel, _ in table.values()}
    w.clone_local(layout, legacy, w.Settings(reserve_bytes=0, log=lambda m: None))
    assert layout.pins() == {f"{s}.czi": h for s, (_, h) in table.items()}
    for slide, (rel, _) in table.items():
        dst = layout.wsi / f"{slide}.czi"
        assert w.is_sealed(dst) and dst.stat().st_ino != before[rel].st_ino
        after = os.stat(legacy / rel)
        assert (after.st_mtime_ns, after.st_mode, after.st_flags) == (
            before[rel].st_mtime_ns, before[rel].st_mode, before[rel].st_flags,
        )  # fmt: skip
    rows = w.jsonl(layout.prov / "local-clones.jsonl")
    assert {r["slide_id"] for r in rows} == set(table)
    assert all(r["method"].startswith("APFS clone") for r in rows)


def test_clone_with_wrong_content_is_refused(
    tmp_path: Path, layout: w.Layout, monkeypatch: pytest.MonkeyPatch
) -> None:
    legacy, table = _fake_local(tmp_path)
    slide = sorted(table)[0]
    table[slide] = (table[slide][0], "0" * 64)
    monkeypatch.setattr(w, "LOCAL", table)
    with pytest.raises(w.Refused, match="!= inventory"):
        w.clone_local(layout, legacy, w.Settings(reserve_bytes=0, log=lambda m: None))
    assert layout.pins() == {}


def test_provenance_covers_all_eight_and_is_write_once(
    ebi: tuple[MockEBI, w.Settings, list[str]],
    tmp_path: Path,
    layout: w.Layout,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, s, _ = ebi
    with pytest.raises(w.Refused, match="0 pins and 0 records"):
        w.provenance(layout)
    legacy, table = _fake_local(tmp_path)
    monkeypatch.setattr(w, "LOCAL", table)
    w.acquire(layout, s)
    w.clone_local(layout, legacy, s)
    out = w.provenance(layout)
    doc = json.loads(out.read_text())
    kinds = {r["slide_id"]: r["kind"] for r in doc["wsi"]}
    assert set(kinds) == pf.PILOT
    assert sorted(kinds.values()) == ["ebi-https"] * 5 + ["local-clone"] * 3
    assert w.provenance(layout) == out  # identical re-run accepted
    assert _gate(pf.g_no_test, layout.data, stage="extract")[0] == "PASS"
    assert _gate(pf.g_immutable, layout.data, stage="extract")[0] == "PASS"


# --------------------------------------------------------------------------- preflight


def test_sums_parser_refuses_malformed_and_duplicate_lines() -> None:
    h = "a" * 64
    assert pf.parse_sums(f"{h}  x.czi\n") == {"x.czi": h}
    for bad in (f"{h} x.czi", f"{h.upper()}  x.czi", f"{h}  x.czi\n{h}  x.czi", f"{h}  "):
        with pytest.raises(ValueError):
            pf.parse_sums(bad)


def _fake_repo(tmp_path: Path) -> Path:
    """Symlinks to the real bound files, so the preflight's own path still resolves."""
    repo = tmp_path / "repo"
    for rel in [*pf.bound_files(), pf.COHORT_DIR / "cohort.yaml"]:
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        (repo / rel).symlink_to(ROOT / rel)
    return repo


def _authorize(repo: Path, summary: str) -> None:
    cfg = (ROOT / pf.COHORT_DIR / "cohort.yaml").read_text()
    entry = (
        "  - id: D-11\n    summary: >-\n      " + summary + "\n    status: approved\n"
        "    gates: [pilot, extraction]\n    approver: test\n    approved_on: '2026-11-02'\n"
        "    reference: test\n"
    )
    target = repo / pf.COHORT_DIR / "cohort.yaml"
    target.unlink()
    target.write_text(cfg.replace("\nlimitations:", entry + "\nlimitations:"))


def test_c2_authorization_is_bound_to_exact_artifact_digests(tmp_path: Path) -> None:
    repo = _fake_repo(tmp_path)
    binding = pf.canonical(pf.binding_document(repo))
    (repo / pf.BINDING_FILE).write_bytes(binding)
    digest = hashlib.sha256(binding).hexdigest()
    text = f"{pf.C2_PHRASE} for pilot selection {pf.SELECTION_ID[:18]}; c2-binding sha256 {digest}"
    _authorize(repo, text)

    def g6(auth: str | None = "D-11") -> tuple[str, str]:
        return pf.g_c2(argparse.Namespace(repo=repo, c2_authorization=auth))

    assert g6()[0] == "PASS", g6()
    assert g6("D-9")[0] == "FAIL" and g6(None)[0] == "FAIL"
    # Any edit to a bound file after authorization fails the gate.
    pilot = repo / pf.COHORT_DIR / "pilot.yaml"
    pilot.unlink()
    pilot.write_text((ROOT / pf.COHORT_DIR / "pilot.yaml").read_text() + "# edited\n")
    status, detail = g6()
    assert status == "FAIL" and "changed since authorization: projects/cohorts" in detail
    # A binding file that is not the authorized one fails before anything else.
    _authorize(repo, text.replace(digest, "f" * 64))
    assert "differs from the authorized" in g6()[1]
    _authorize(repo, f"{pf.C2_PHRASE} for {pf.SELECTION_ID[:18]}")
    assert "does not name 'c2-binding sha256" in g6()[1]


def test_binding_names_every_executable_input() -> None:
    doc = pf.binding_document(ROOT)
    assert set(doc["files"]) == {p.as_posix() for p in pf.bound_files()}
    assert doc["downloads"] == w.ALLOW and doc["uni_checkpoint_sha256"] == pf.UNI_SHA256
    assert {
        "projects/cohorts/surgen-msi/c2-pilot/c2_wsi.py",
        "pipelines/wsi-patch-embedding-uni.v1.yaml",
    } <= set(doc["files"])


# --------------------------------------------------------------------------- re-selection


def _retake(tmp_path: Path, extra: dict[str, str] | None = None) -> tuple[str, list[str]]:
    """Inventory over pilot-wsi (plus `extra` slide -> root), re-select, check equivalence."""
    from domain.pilot import select_pilot
    from domain.wsi_inventory import (
        Access,
        InventoryRoot,
        LocationKind,
        ObservedObject,
        Verification,
        build_inventory,
    )
    from infrastructure.cohort_sources import read_manifest
    from infrastructure.wsi_inventory import load_pilot_criteria, write_inventory, write_pilot

    manifest = read_manifest(ROOT / pf.COHORT_DIR / "manifests/v1/cohort-manifest.json")
    pins = {f"{s}.czi": h for s, (_, h) in pf.LOCAL.items()}
    pins |= {f"{s}.czi": hashlib.sha256(s.encode()).hexdigest() for s in pf.PENDING}
    located = {name: pf.RETAKEN_ROOT for name in pins}
    located |= {f"{s}.czi": root for s, root in (extra or {}).items()}
    objects = [
        ObservedObject(
            root, name, LocationKind.LOCAL, Access.READABLE, Verification.CONTENT_VERIFIED,
            1000, pins.get(name) or hashlib.sha256(name.encode()).hexdigest(),
        )
        for name, root in sorted(located.items())
    ]  # fmt: skip
    roots = [
        InventoryRoot(r, LocationKind.LOCAL, "local directory bound at run time")
        for r in sorted(set(located.values()))
    ]
    inventory = build_inventory(manifest, roots, ["*.czi"], objects)
    criteria = load_pilot_criteria(ROOT / pf.COHORT_DIR / "pilot.yaml")
    selection = select_pilot(manifest, criteria, inventory)
    inv_path = write_inventory(tmp_path / "inventories", inventory)
    sel_path = write_pilot(tmp_path / "pilots", selection)
    return pf.retaken_problems(ROOT, sel_path, inv_path, pins)


def test_retaken_selection_over_the_pilot_root_is_the_approved_pilot(tmp_path: Path) -> None:
    new_id, problems = _retake(tmp_path)
    assert problems == []
    assert new_id != pf.SELECTION_ID  # the id always changes: equivalence is by content


def test_an_extra_verified_slide_elsewhere_changes_the_retaken_selection(tmp_path: Path) -> None:
    """Why the re-inventory binds only the pilot-wsi root: verified-first ranking means one
    more SurGen file in ~/Downloads can displace an approved slide."""
    from domain.pilot import rank_key
    from infrastructure.cohort_sources import read_manifest

    manifest = read_manifest(ROOT / pf.COHORT_DIR / "manifests/v1/cohort-manifest.json")
    approved = json.loads(
        (ROOT / pf.COHORT_DIR / "pilots" / pf.SELECTION_ID / "pilot-selection.json").read_text()
    )["body"]
    seed = approved["criteria"]["seed"]
    cutoff = max(s["rank_key"] for s in approved["slides"] if s["stratum"] == ["SR386", "negative"])
    intruder = min(
        (r for r in manifest.eligible()
         if (r.cohort, r.label) == ("SR386", "negative") and r.partition != "test"
         and r.slide_id not in pf.PILOT and rank_key(seed, r.slide_id) < cutoff),
        key=lambda r: rank_key(seed, r.slide_id),
    )  # fmt: skip
    _, problems = _retake(tmp_path, {intruder.slide_id: "downloads"})
    assert "slides differ from the approved selection" in problems
    assert any("inventory roots" in p for p in problems)


def test_g13_is_unresolved_without_the_retaken_documents(tmp_path: Path) -> None:
    a = argparse.Namespace(
        repo=ROOT, data_root=tmp_path, stage="extract", retaken_selection=None,
        retaken_inventory=None,
    )  # fmt: skip
    assert pf.g_retaken(a)[0] == "UNRESOLVED"


def test_preflight_extract_stage_refuses_leftover_staging(tmp_path: Path) -> None:
    stage = tmp_path / "staging" / "ebi"
    stage.mkdir(parents=True)
    (stage / f"{pf.PENDING[0]}.czi.transfer.json").write_text("{}")
    assert _gate(pf.g_no_test, tmp_path)[0] == "PASS"
    assert _gate(pf.g_no_test, tmp_path, stage="extract")[0] == "FAIL"
    (stage / "notes.txt").write_text("")
    assert _gate(pf.g_no_test, tmp_path)[0] == "FAIL"


def test_sealed_requires_the_immutable_flag_for_wsis(tmp_path: Path) -> None:
    f = tmp_path / "x.czi"
    f.write_bytes(b"x")
    os.chmod(f, 0o444)
    assert not pf.sealed(f)
    os.chflags(f, stat.UF_IMMUTABLE)
    try:
        assert pf.sealed(f) and w.is_sealed(f)
    finally:
        os.chflags(f, 0)
