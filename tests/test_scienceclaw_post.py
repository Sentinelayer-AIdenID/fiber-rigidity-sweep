from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading

import pytest

from run_sweep import run
from scienceclaw import post_results as poster
from scienceclaw.results import load_report


ROOT = Path(__file__).resolve().parents[1]
# Inert local-server sentinels, never credentials for any real service.
API_VALUE = "local-test-api-value"
SESSION_VALUE = "local-test-session-value"


@pytest.fixture(scope="module")
def finished(tmp_path_factory):
    output = tmp_path_factory.mktemp("completed-sweep")
    run(dict(nx=6, ny=6, z=[3.8], kappa=[.01, .03], gamma=[.001, .01, .1], seeds=[3],
             backend="numpy", fire=dict(max_steps=0), output=str(output),
             batch_size=3, viewer_max_edges=10))
    return output


@contextmanager
def fake_server(*, login_status=200, post_status=201, login_body=None, post_body=None):
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append((self.path, body, self.headers.get("Authorization")))
            login = self.path == "/api/agents/login"
            status = login_status if login else post_status
            data = (login_body if login_body is not None else {"token": SESSION_VALUE}) if login else (
                post_body if post_body is not None else {"post": {"id": "local-post-17"}})
            if status >= 400:
                data = {"error": "Minimum 10 karma required to post", "echo": [API_VALUE, SESSION_VALUE]}
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            if status in (301, 302, 307, 308):
                self.send_header("Location", "/api/redirect-target")
            self.end_headers()
            self.wfile.write(json.dumps(data).encode())

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": .01}, daemon=True)
    worker.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/api", requests
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=2)


def base_args(finished):
    return ["--out", str(finished), "--community", "materials"]


def assert_no_secrets(captured):
    assert API_VALUE not in captured.out + captured.err
    assert SESSION_VALUE not in captured.out + captured.err


def test_dry_run_default_no_network(finished, monkeypatch, capsys, tmp_path):
    def forbidden(*args, **kwargs):
        pytest.fail("dry run attempted credentials or network")

    monkeypatch.setattr(poster, "api_request", forbidden)
    monkeypatch.setattr(poster, "load_api_key", forbidden)
    draft = tmp_path / "draft.json"
    assert poster.main(base_args(finished) + ["--draft-out", str(draft),
        "--repo-url", "https://example.org/repo", "--viewer-url", "https://example.org/viewer"]) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    payload = json.loads(captured.out)
    assert json.loads(draft.read_text()) == payload
    assert {"community", "title", "content", "hypothesis", "method", "findings", "dataSources",
            "openQuestions", "toolsUsed", "evidenceSummary"} <= payload.keys()
    assert len(payload["title"]) <= 300
    assert payload["dataSources"] == ["https://example.org/repo", "https://example.org/viewer"]
    assert payload["toolsUsed"] == ["fiber-network-rigidity"]
    assert "0/6 states converged" in payload["findings"]
    assert "not determined" in payload["content"]
    assert "| 3.8 |" in payload["content"]
    assert "conditional" in payload["content"]
    assert "not reproductions of published" in payload["content"]
    assert_no_secrets(captured)


@pytest.mark.parametrize("source", ["api_key", "apiKey", "env"])
def test_post_success_no_secret_output(finished, monkeypatch, capsys, source, tmp_path):
    if source == "env":
        monkeypatch.setenv("LOCAL_TEST_API", API_VALUE)
        credentials = ["--api-key-env", "LOCAL_TEST_API"]
    else:
        # Exercise config parsing without persisting credential-shaped material.
        config = tmp_path / "credentials.json"
        original = Path.read_text
        monkeypatch.setattr(Path, "read_text", lambda self, *a, **kw:
            json.dumps({source: API_VALUE}) if self == config else original(self, *a, **kw))
        credentials = ["--config", str(config)]
    with fake_server() as (url, requests):
        monkeypatch.setenv("INFINITE_API_BASE", url)
        api_args = [] if source == "env" else ["--api-base", url]
        assert poster.main(base_args(finished) + ["--post"] + api_args + credentials) == 0
    captured = capsys.readouterr()
    assert_no_secrets(captured)
    assert captured.err == ""
    assert json.loads(captured.out) == {"post_id": "local-post-17"}
    assert len(requests) == 2
    assert requests[0] == ("/api/agents/login", {"apiKey": API_VALUE}, None)
    path, payload, auth = requests[1]
    assert path == "/api/posts" and auth == "Bearer " + SESSION_VALUE
    assert payload == poster.build_payload(load_report(finished, ROOT), "materials")


@pytest.mark.parametrize("options,message,count", [
    ({"login_status": 401}, "HTTP 401", 1),
    ({"post_status": 403}, "at least 10 karma", 2),
    ({"post_status": 500}, "HTTP 500", 2),
    ({"login_body": {}}, "no valid token", 1),
    ({"login_body": []}, "invalid JSON object", 1),
    ({"post_body": {}}, "publication is unconfirmed", 2),
    ({"post_body": {"id": API_VALUE}}, "unsafe post id", 2),
    ({"post_body": {"id": SESSION_VALUE}}, "unsafe post id", 2),
    ({"login_status": 302}, "HTTP 302", 1),
    ({"post_status": 307}, "HTTP 307", 2),
])
def test_post_errors_fail_closed(finished, monkeypatch, capsys, options, message, count):
    monkeypatch.setenv("LOCAL_TEST_API", API_VALUE)
    with fake_server(**options) as (url, requests):
        assert poster.main(base_args(finished) + ["--post", "--api-base", url,
                                                 "--api-key-env", "LOCAL_TEST_API"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert message in captured.err
    assert_no_secrets(captured)
    assert len(requests) == count


@pytest.mark.parametrize("credential_args", [[], ["--config", "/missing/infinite-config.json"],
                                            ["--api-key-env", "ABSENT_TEST_VARIABLE"]])
def test_missing_credentials_fail_before_network(finished, monkeypatch, capsys, credential_args):
    monkeypatch.delenv("ABSENT_TEST_VARIABLE", raising=False)
    monkeypatch.setattr(poster, "api_request", lambda *a, **k: pytest.fail("unexpected network"))
    assert poster.main(base_args(finished) + ["--post"] + credential_args) == 1
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err
    assert_no_secrets(captured)


@pytest.mark.parametrize("document", ["invalid JSON", "[]", "{}", '{"api_key": 7}'])
def test_invalid_credentials(finished, monkeypatch, capsys, tmp_path, document):
    path = tmp_path / "invalid-config.json"
    path.write_text(document)
    monkeypatch.setattr(poster, "api_request", lambda *a, **k: pytest.fail("unexpected network"))
    assert poster.main(base_args(finished) + ["--post", "--config", str(path)]) == 1
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err


def test_artifact_mismatch_fails(finished, tmp_path, capsys):
    import shutil
    output = tmp_path / "mismatch"
    shutil.copytree(finished, output)
    path = output / "manifest.json"
    document = json.loads(path.read_text())
    document["fingerprint"] = "different-run"
    path.write_text(json.dumps(document))
    assert poster.main(base_args(output)) == 1
    captured = capsys.readouterr()
    assert captured.out == "" and "completed sweep" in captured.err


def test_computed_fit_is_reported_with_uncertainty(finished):
    report = load_report(finished, ROOT)
    # A reporting fixture, not a measured scientific result.
    row = report["results"][0]
    row["gamma_c_modulus"] = .125
    row["critical_exponents"]["gamma_c_K"].update(status="determined", f=.75,
        f_stderr=.02, f_ci95=[.70, .80], fit_window=[.14, .2], n_points=6)
    content = poster.build_payload(report, "materials")["content"]
    assert "0.125" in content and "f=0.75 ± 0.02" in content
    assert "95% CI [0.7, 0.8]" in content and "strain window [0.14, 0.2]" in content
    assert "6 points" in content


@pytest.mark.parametrize("base", ["http://example.org/api", "https://user:password@example.org/api",
                                "https://example.org/api?key=anything", "https://[invalid"])
def test_unsafe_api_base_fails_before_network(finished, monkeypatch, capsys, base):
    monkeypatch.setenv("LOCAL_TEST_API", API_VALUE)
    monkeypatch.setattr(poster, "api_request", lambda *a, **k: pytest.fail("unexpected network"))
    assert poster.main(base_args(finished) + ["--post", "--api-key-env", "LOCAL_TEST_API",
                                             "--api-base", base]) == 1
    captured = capsys.readouterr()
    assert captured.out == "" and "HTTPS" in captured.err
    assert_no_secrets(captured)


def test_transport_error_never_echoes_secret(finished, monkeypatch, capsys):
    import urllib.error

    class BrokenOpener:
        def open(self, *args, **kwargs):
            raise urllib.error.URLError(API_VALUE + SESSION_VALUE)

    monkeypatch.setenv("LOCAL_TEST_API", API_VALUE)
    monkeypatch.setattr(poster.urllib.request, "build_opener", lambda *a: BrokenOpener())
    assert poster.main(base_args(finished) + ["--post", "--api-key-env", "LOCAL_TEST_API"]) == 1
    captured = capsys.readouterr()
    assert captured.out == "" and "request failed" in captured.err
    assert_no_secrets(captured)


def test_credential_in_draft_is_not_written(finished, monkeypatch, capsys, tmp_path):
    monkeypatch.setenv("LOCAL_TEST_API", API_VALUE)
    monkeypatch.setattr(poster, "api_request", lambda *a, **k: pytest.fail("unexpected network"))
    draft = tmp_path / "unsafe-draft.json"
    assert poster.main(base_args(finished) + ["--post", "--api-key-env", "LOCAL_TEST_API",
        "--repo-url", "https://example.org/" + API_VALUE, "--draft-out", str(draft)]) == 1
    captured = capsys.readouterr()
    assert captured.out == "" and not draft.exists()
    assert_no_secrets(captured)
