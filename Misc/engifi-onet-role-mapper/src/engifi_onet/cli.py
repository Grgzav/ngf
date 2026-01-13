from __future__ import annotations

import typer
from rich import print as rprint

from .config import settings
from .http import OnetHttpClient

app = typer.Typer(add_completion=False, help="Engifi O*NET role mapper (repo scaffold).")


@app.command()
def info() -> None:
    """Print current configuration (does not reveal the API key)."""
    rprint("[bold]engifi-onet-role-mapper[/bold]")
    rprint(f"ONET_BASE_URL: {settings.onet_base_url}")
    rprint(f"ONET_API_KEY set: {bool(settings.onet_api_key)}")


@app.command()
def ping() -> None:
    """Minimal live call (requires ONET_API_KEY): keyword search."""
    client = OnetHttpClient()
    data = client.get_json("/online/search", params={"keyword": "engineer"})
    total = data.get("total", None)
    first = (data.get("occupation", []) or [None])[0]
    rprint(f"[green]OK[/green] total={total}")
    if first:
        rprint(f"Top result: {first.get('title')} ({first.get('code')})")


if __name__ == "__main__":
    app()