#!/usr/bin/env python3
"""Build an Infinite post from a finished sweep; network posting is opt-in."""

import argparse
import http.client
import json
import os
from pathlib import Path
import sys
import urllib.error
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

from scienceclaw.results import load_report

DEFAULT_API_BASE = "https://infinite-lamm.vercel.app/api"
FORBIDDEN_CODES = {
    "AGENT_BANNED": "The forum reports that this agent is banned.",
    "INSUFFICIENT_KARMA": "The forum requires more karma for this action.",
}


class PostError(Exception):
    """A safe-to-display error; never includes raw credentials or server data."""


def number(value):
    return format(value, ".6g") if isinstance(value, (int, float)) else "not determined"


def build_payload(report, community, repo_url=None, viewer_url=None):
    parameters = report["parameters"]
    method = (f"Independent affine-start FIRE minimizations; backend {parameters['backend']}; "
              f"{parameters['nx']} × {parameters['ny']} precursor sites; "
              f"{len(parameters['seeds'])} seeds and {len(parameters['gamma'])} strains. "
              "Critical strains and conditional fits use convergence-filtered mean curves.")
    table = ["| Target z | Measured z | κ | γc modulus | γc nonaffinity | Agreement log10 | Converged |",
             "|---|---|---|---|---|---|---|"]
    fits = []
    for row in report["results"]:
        values = [number(row[key]) for key in ("z", "measured_z", "kappa", "gamma_c_modulus",
                                               "gamma_c_nonaffinity", "agreement_log10")]
        table.append("| " + " | ".join(values + [f"{row['converged']}/{row['total']}"]) + " |")
        for name, fit in row["critical_exponents"].items():
            label = f"z={number(row['measured_z'])}, κ={number(row['kappa'])}, {name}"
            if fit["status"] == "determined":
                text = (f"f={number(fit['f'])} ± {number(fit['f_stderr'])} (standard error); "
                        f"95% CI {fit['f_ci95']}; strain window {fit['fit_window']}; "
                        f"{fit['n_points']} points")
            else:
                text = f"f: not determined ({fit['reason']}); {fit['n_points']} usable points"
            fits.append(f"- {label}: {text}.")
    findings = report["summary"] + "\n\n" + "\n".join(table) + "\n\n" + "\n".join(fits)
    content = ("## Sweep\n\n" + method + "\n\n## Computed results\n\n" + findings
               + "\n\n## Limitations\n\n" + report["limitations"])
    return dict(community=community, title=f"Fiber-network rigidity sweep: {len(report['results'])} curves",
                content=content, hypothesis="Test whether strain stiffening produces resolvable critical onsets in this sweep.",
                method=method, findings=findings, dataSources=[url for url in (repo_url, viewer_url) if url],
                openQuestions=["How do larger networks, more seeds and denser strain grids change the onsets and fitted exponents?"],
                toolsUsed=["fiber-network-rigidity"], evidenceSummary=report["summary"])


def load_post_settings(config, env_name, base_override):
    base = DEFAULT_API_BASE
    if env_name:
        value = os.environ.get(env_name)
    elif config:
        try:
            document = json.loads(Path(config).expanduser().read_text(encoding="utf-8"))
        except (OSError, ValueError):
            raise PostError("Cannot read valid Infinite credentials JSON.") from None
        if not isinstance(document, dict):
            raise PostError("Infinite credentials must be a JSON object.")
        value = document.get("api_key") or document.get("apiKey")
        base = document.get("api_base", DEFAULT_API_BASE)
    else:
        raise PostError("--post requires --config or --api-key-env with your own Infinite credentials.")
    if not isinstance(value, str) or not value.strip():
        raise PostError("Infinite API key is missing or invalid.")
    base = base_override if base_override is not None else base
    return value, validate_api_base(base, allow_local_http=base_override is not None)


def validate_api_base(base, *, allow_local_http=False):
    try:
        if not isinstance(base, str) or any(c.isspace() or ord(c) < 32 for c in base):
            raise ValueError
        url = urllib.parse.urlsplit(base)
        local = url.hostname in ("localhost", "127.0.0.1", "::1")
        if (url.scheme != "https" and not (allow_local_http and url.scheme == "http" and local)
                or not url.hostname or url.username is not None or url.password is not None
                or "?" in base or "#" in base):
            raise ValueError
        _ = url.port
    except ValueError:
        raise PostError("API base must be an absolute HTTPS URL without credentials, query or fragment "
                        "(explicit overrides allow HTTP only for local testing).") from None
    return base.rstrip("/")


def forbidden_message(exc, endpoint):
    action = "post" if endpoint == "/posts" else "request"
    message = (f"The forum refused the {action} (HTTP 403); "
               "check your agent's standing and the community's rules.")
    try:
        raw = exc.read(4097)
        document = json.loads(raw) if len(raw) <= 4096 else None
        code = document.get("code") if isinstance(document, dict) else None
        if isinstance(code, str) and code in FORBIDDEN_CODES:
            message += " " + FORBIDDEN_CODES[code]
    except (OSError, ValueError, http.client.HTTPException):
        pass
    return message


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Login bodies and Bearer headers must not travel to another endpoint.
        return None


def api_request(base, endpoint, body, token=None):
    headers = {"Content-Type": "application/json"}
    if token is not None:
        headers["Authorization"] = "Bearer " + token
    request = urllib.request.Request(base + endpoint, data=json.dumps(body).encode("utf-8"),
                                     headers=headers, method="POST")
    try:
        opener = urllib.request.build_opener(NoRedirect())
        with opener.open(request, timeout=30) as response:
            if response.status != 200 and not (endpoint == "/posts" and response.status == 201):
                raise PostError("Infinite returned an unexpected HTTP status.")
            raw = response.read(1_048_577)
        if len(raw) > 1_048_576:
            raise PostError("Infinite response is too large.")
        result = json.loads(raw)
        if not isinstance(result, dict):
            raise PostError("Infinite returned an invalid JSON object.")
        return result
    except urllib.error.HTTPError as exc:
        if exc.code == 403:
            message = forbidden_message(exc, endpoint)
        elif exc.code == 401:
            message = "Infinite authentication failed (HTTP 401)."
        else:
            message = f"Infinite request failed (HTTP {exc.code})."
        exc.close()
        raise PostError(message) from None
    except (urllib.error.URLError, OSError, ValueError, http.client.HTTPException):
        # Exception strings, redirect locations and response bodies can echo secrets.
        raise PostError("Infinite request failed or returned invalid JSON.") from None


def publish(payload, base, key):
    base = validate_api_base(base, allow_local_http=True)
    login = api_request(base, "/agents/login", {"apiKey": key})
    token = login.get("token")
    if not isinstance(token, str) or not token or any(c.isspace() for c in token):
        raise PostError("Infinite login returned no valid token.")
    result = api_request(base, "/posts", payload, token)
    post = result.get("post", result)
    post_id = post.get("id") if isinstance(post, dict) else None
    if (not isinstance(post_id, (str, int)) or isinstance(post_id, bool)
            or not str(post_id) or len(str(post_id)) > 300):
        raise PostError("Infinite did not return a post id; publication is unconfirmed.")
    if any(secret in str(post_id) for secret in (key, token)):
        raise PostError("Infinite returned an unsafe post id; publication is unconfirmed.")
    return post_id


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="Finished sweep output directory")
    parser.add_argument("--community", required=True)
    parser.add_argument("--repo-url")
    parser.add_argument("--viewer-url")
    parser.add_argument("--draft-out", help="Write the exact JSON payload to this file")
    parser.add_argument("--post", action="store_true", help="Explicitly publish using your own account")
    credentials = parser.add_mutually_exclusive_group()
    credentials.add_argument("--config", help="Your Infinite credentials JSON (not a sweep YAML)")
    credentials.add_argument("--api-key-env", help="Name of an environment variable containing your API key")
    parser.add_argument("--api-base", default=os.environ.get("INFINITE_API_BASE"))
    args = parser.parse_args(argv)
    try:
        report = load_report(args.out, ROOT)
        payload = build_payload(report, args.community, args.repo_url, args.viewer_url)
        draft = json.dumps(payload, indent=2, allow_nan=False)
        # Dry runs neither read credentials nor construct an HTTP client.
        key, base = load_post_settings(args.config, args.api_key_env, args.api_base) if args.post else (None, None)
        if key and key in draft:
            raise PostError("The draft contains credential material; refusing to output or post it.")
        if args.draft_out:
            Path(args.draft_out).write_text(draft + "\n", encoding="utf-8")
        if args.post:
            post_id = publish(payload, base, key)
            print(json.dumps({"post_id": post_id}))
        else:
            print(draft)
    except PostError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except (OSError, ValueError, KeyError, TypeError, IndexError):
        print("Cannot read a completed sweep or write the draft; check paths and artifact schemas.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
