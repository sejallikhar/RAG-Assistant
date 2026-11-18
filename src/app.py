"""CLI.  python -m src.app ingest | ask "..." | predict --rank N --category C --state S | chat"""
import argparse
import json

from . import ingest, config
from .pipeline import answer
from .predictor import predict, format_result


def show(res):
    print(f"[{res.status.upper()}]" + (f" ({', '.join(res.reasons)})" if res.reasons else ""))
    print(res.answer)
    if res.citations:
        print("\nSources:")
        for c in res.citations:
            print(f"  [{c['label']}] {c['id']}")


def main():
    ap = argparse.ArgumentParser(prog="src.app")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ingest")
    a = sub.add_parser("ask"); a.add_argument("question")
    sub.add_parser("chat")
    p = sub.add_parser("predict")
    p.add_argument("--rank", type=int, required=True); p.add_argument("--category", required=True)
    p.add_argument("--state", required=True); p.add_argument("--course", default="MBBS")
    p.add_argument("--year", type=int); p.add_argument("--json", action="store_true")
    args = ap.parse_args()

    if args.cmd == "ingest":
        ingest.build()
    elif args.cmd == "ask":
        show(answer(args.question))
    elif args.cmd == "chat":
        print(f"backend={config.LLM_BACKEND} model={config.LLM_MODEL}  (empty line to quit)")
        while (q := input("\n> ").strip()):
            show(answer(q))
    elif args.cmd == "predict":
        res = predict(args.rank, args.category, args.state, args.course, args.year)
        print(json.dumps(res, indent=2) if args.json else format_result(res))


if __name__ == "__main__":
    main()
