"""`adapdomain verify-remote` implementation."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import typer

from adapdomain.backends.ascend.healthcheck import run_healthcheck
from adapdomain.storage.run_manifest import RunManifest

app = typer.Typer(help="Verify CANN/ACL environment on the remote NPU host.")


@app.callback(invoke_without_command=True)
def verify(
    smoke_om: str | None = typer.Option(None, help="Optional .om path for smoke inference"),
    output_root: str = typer.Option("data/results/healthcheck", help="Result directory"),
) -> None:
    run_id = f"healthcheck-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    out_dir = Path(output_root) / run_id
    manifest = RunManifest.start(run_id=run_id, command="adapdomain verify-remote", out_dir=out_dir)
    rep = run_healthcheck(smoke_om_path=smoke_om)
    (out_dir / "report.json").write_text(json.dumps(rep.to_dict(), indent=2, sort_keys=True))
    manifest.set_hardware(
        "ascend_910b3" if rep.device_count else "unknown",
        device_count=rep.device_count or 0,
        cann_version=None,
    )
    manifest.set_status("done" if rep.acl_importable else "error")
    manifest.finalize()
    typer.echo(out_dir / "report.json")
