#!/usr/bin/env python
"""Print the exact Hugging Face commit of every configured embedding model.

Runs record the resolved commit automatically (``model_revisions`` in each run's
``config.json``).  For a frozen evaluation, copy the printed values into the
``revision`` / ``query_revision`` fields of the config so later runs cannot drift.

    python scripts/pin_model_revisions.py configs/synthetic.yaml
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.core.config import load_app_config
from app.embeddings.base import resolve_commit_hash


def main() -> None:
    config = load_app_config(sys.argv[1] if len(sys.argv) > 1 else "synthetic.yaml")
    for key, emb in sorted(config.embeddings.items()):
        if emb.provider == "hashing":
            continue
        print(f"{key}:")
        print(f"  revision: {resolve_commit_hash(emb.model_id, emb.revision)}   # {emb.model_id}")
        if emb.query_model_id:
            print(f"  query_revision: {resolve_commit_hash(emb.query_model_id, emb.query_revision)}"
                  f"   # {emb.query_model_id}")


if __name__ == "__main__":
    main()
