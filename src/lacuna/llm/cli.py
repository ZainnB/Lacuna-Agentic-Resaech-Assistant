"""`lacuna llm status` and `lacuna llm ping`."""

from __future__ import annotations

import json

from lacuna.llm.config import load_config
from lacuna.llm.providers import PROVIDERS, split_spec, with_overrides
from lacuna.settings import get_settings


def cmd_status(args) -> None:
    settings = get_settings()
    cfg = load_config()
    usage: dict[str, list[dict]] = {}
    note = ""
    try:
        from lacuna.db import connect, quota_day

        with connect() as conn:
            rows = conn.execute(
                "SELECT * FROM provider_usage WHERE day >= %s ORDER BY provider, model",
                (quota_day("gemini"),),
            ).fetchall()
        for r in rows:
            usage.setdefault(r["provider"], []).append(r)
    except Exception as e:  # noqa: BLE001
        note = f"(usage unavailable: Postgres not reachable: {type(e).__name__})"
    print(f"spend_stage={settings.spend_stage}")
    print(f"{'provider':<11}{'key':<5}{'paid':<6}{'rpm':>5}{'rpd':>7}{'tpm':>8}{'tpd':>9}  today")
    for name, base in PROVIDERS.items():
        p = with_overrides(base, cfg.limits.get(name))
        key = "yes" if p.available(settings) else "no"
        if p.key_attr is None and not p.paid:
            key = "n/a"
        today = ", ".join(
            f"{u['model']}: calls={u['calls']} tok={u['input_tokens']}+{u['output_tokens']} "
            f"err={u['errors']} 429={u['rate_limited']} fb={u['fallbacks']} "
            f"schema_fail={u['schema_failures']}"
            for u in usage.get(name, [])
        )
        print(
            f"{name:<11}{key:<5}{'yes' if p.paid else 'no':<6}{p.rpm or '-':>5}{p.rpd or '-':>7}"
            f"{p.tpm or '-':>8}{p.tpd or '-':>9}  {today or '-'}"
        )
    print("chains:")
    for role, chain in cfg.chains.items():
        print(f"  {role}: {' -> '.join(chain)}")
    if note:
        print(note)


def cmd_ping(args) -> None:
    from lacuna.llm.chain import complete_json

    split_spec(args.spec)
    schema = {
        "type": "object",
        "properties": {"ok": {"type": "boolean"}},
        "required": ["ok"],
        "additionalProperties": False,
    }
    r = complete_json(
        "default",
        "You reply with JSON only.",
        'Reply with {"ok": true}.',
        schema,
        chain=[args.spec],
        max_tokens=64,
        use_cache=False,
    )
    print(
        json.dumps(
            {
                "spec": r.spec,
                "parsed": r.parsed,
                "tokens": [r.input_tokens, r.output_tokens],
                "latency_ms": r.latency_ms,
                "stop": r.stop_reason,
            }
        )
    )


def register(sub) -> None:
    p = sub.add_parser("llm", help="LLM provider layer")
    s = p.add_subparsers(dest="llm_cmd", required=True)
    s.add_parser("status", help="providers, keys, limits, today's usage").set_defaults(fn=cmd_status)
    pg = s.add_parser("ping", help="one live call to a provider (needs its key)")
    pg.add_argument("--spec", required=True, help="e.g. groq/openai/gpt-oss-120b")
    pg.set_defaults(fn=cmd_ping)
