"""Use TuGraph's REST API to verify imported vertex and edge totals."""
import argparse
import json
import os
from pathlib import Path
from urllib.request import Request, urlopen


def post(url, body, token=None):
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = "Bearer " + token
    request = Request(url, data=json.dumps(body).encode("utf-8"), headers=headers, method="POST")
    with urlopen(request, timeout=120) as response:
        return json.loads(response.read().decode("utf-8"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:7071")
    parser.add_argument("--graph", default="default")
    parser.add_argument("--user", default=os.environ.get("TUGRAPH_USER", "admin"))
    parser.add_argument("--password", default=os.environ.get("TUGRAPH_PASSWORD"))
    args = parser.parse_args()
    if not args.password:
        parser.error("Set TUGRAPH_PASSWORD or provide --password.")
    login = post(args.base_url.rstrip("/") + "/login", {"user": args.user, "password": args.password})
    token = login["jwt"]
    counts = {}
    for name, query in [
        ("webpage_count", "MATCH (n:webpage) RETURN count(n) AS n"),
        ("hyperlink_count", "MATCH ()-[r:hyperlink]->() RETURN count(r) AS n"),
    ]:
        response = post(args.base_url.rstrip("/") + "/cypher", {"graph": args.graph, "script": query}, token)
        counts[name] = response["result"][0][0]
    result = {"base_url": args.base_url, "graph": args.graph, **counts}
    output = Path(__file__).resolve().parents[1] / "results" / "tugraph_verification.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
