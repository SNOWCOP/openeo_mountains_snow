#%%

#!/usr/bin/env python3
"""Generate the ICAB Mermaid sequence diagram used in WAE architecture docs."""

from __future__ import annotations

import argparse
from pathlib import Path

DIAGRAM = """sequenceDiagram
    participant User
    participant WAEid as WAE Identity (federated via EGI Check-in)
    participant ICAB as Identity & Credit-Awareness Broker
    participant Vault as Credential Vault
    participant CDSE
    participant DESP as DESP/DEDL
    participant Orch as Orchestration Layer

    User->>WAEid: Login once (institutional SSO)
    User->>ICAB: "Link CDSE account" (one-time, OAuth2)
    ICAB->>CDSE: OAuth2 authorize + token exchange
    CDSE-->>ICAB: scoped refresh token
    ICAB->>Vault: store encrypted token (user, CDSE)
    User->>ICAB: "Link DESP/DEDL account" (one-time, OIDC)
    ICAB->>DESP: OIDC login via destine_auth
    DESP-->>ICAB: access + refresh token
    ICAB->>Vault: store encrypted token (user, DESP)

    Note over ICAB: Background: poll credit/quota per linked account

    Orch->>ICAB: "Which platforms can serve Sentinel-2 + have budget?"
    ICAB->>CDSE: check openEO credit usage (cached/live)
    ICAB->>DESP: check DEDL service quota (cached/live)
    ICAB-->>Orch: {CDSE: 40 credits left, DESP: quota exhausted}
    Orch->>ICAB: execute on CDSE using stored token (silent)
    ICAB->>Vault: fetch + refresh token
    ICAB->>CDSE: authenticated openEO job execution
    CDSE-->>Orch: result + updated credit consumption
    Orch->>ICAB: log consumption to Ledger
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the ICAB Mermaid sequence diagram text."
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Optional output file path (.mmd or .md). If omitted, prints to stdout.",
    )
    # In notebooks, Jupyter adds extra CLI args (for example: --f=...).
    # parse_known_args keeps our script usable in both terminal and notebook runs.
    args, _ = parser.parse_known_args()
    return args


def main() -> None:
    args = parse_args()
    if args.output:
        args.output.write_text(DIAGRAM + "\n", encoding="utf-8")
        print(f"Wrote Mermaid diagram to {args.output}")
    else:
        print(DIAGRAM)


if __name__ == "__main__":
    main()

# %%
