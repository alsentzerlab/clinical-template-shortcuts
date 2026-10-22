"""Baseline reformat of a note pool for the synthetic-bias study.

Streams the GCS parquet in-memory (no raw PHI to disk), excludes any note whose
text contains --exclude_phrase (a string unique to one template, used to hold that template out), samples --n notes, and reformats each ONCE into the standardized
schema (Step 1). Step 2 (deterministic A/B via restructure/make_exact_arms) is
run separately and is free.

Reuses pipeline.process_one so behavior matches the validated pipeline.
Resume-safe: re-run with the same --output to continue from the checkpoint.

Usage:
  python run_pool_reformat.py \
    --gcs_parquet gs://<bucket>/synthetic_source_notes.parquet \
    --exclude_phrase <phrase> --n 1000 --seed 42 \
    --output pool_reformat_1000.csv \
    --model gpt-5-5 --repair --max_workers 4 --check_keys
"""
import argparse, csv, json, logging, os, sys, threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from llm_client import load_keys, validate_keys
from pipeline import load_gcs_parquet, load_schema, process_one, CSV_COLUMNS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
csv.field_size_limit(sys.maxsize)


def load_done(output_path: str) -> set:
    """note_csn_ids already successfully reformatted (resume support)."""
    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        return set()
    try:
        d = pd.read_csv(output_path)
        ok = d[d["reformat_success"] == True]
        return set(ok["note_csn_id"].astype(str))
    except Exception:
        return set()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gcs_parquet", required=True)
    ap.add_argument("--exclude_phrase", required=True, help="Case-insensitive substring; notes containing it are dropped")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--exclude_ids_from", nargs="+", default=None,
                    help="Prior reformat CSV(s); their successful note_csn_ids are removed "
                         "from the sampling pool so a new draw never overlaps them.")
    ap.add_argument("--output", required=True)
    ap.add_argument("--schema", default=None, help="Reformat target schema (default: standardized_template.json)")
    ap.add_argument("--model", default="gpt-5-5")
    ap.add_argument("--repair", action="store_true")
    ap.add_argument("--max_tokens", type=int, default=16000)
    ap.add_argument("--max_workers", type=int, default=4)
    ap.add_argument("--env_file", default=None)
    ap.add_argument("--api_key", default=None)
    ap.add_argument("--check_keys", action="store_true")
    args = ap.parse_args()

    api_key = args.api_key or load_keys(args.env_file)
    if not api_key:
        sys.exit("No API keys found.")
    if args.check_keys:
        good = validate_keys([k.strip() for k in api_key.split(",") if k.strip()], model=args.model)
        if not good:
            sys.exit("All API keys failed validation (401).")
        api_key = ",".join(good)
        logger.info(f"{len(good)} key(s) passed validation")

    schema = load_schema(args.schema) if args.schema else load_schema()

    df = load_gcs_parquet(args.gcs_parquet)
    phrase = args.exclude_phrase.lower()
    is_excl = df["full_note"].astype(str).str.lower().str.contains(phrase, regex=False)
    elig = df[~is_excl]
    logger.info(f"total={len(df)} excluded(phrase '{args.exclude_phrase}')={int(is_excl.sum())} eligible={len(elig)}")

    if args.exclude_ids_from:
        prior = set()
        for pth in args.exclude_ids_from:
            try:
                pdf = pd.read_csv(pth)
                ok = pdf[pdf["reformat_success"] == True] if "reformat_success" in pdf.columns else pdf
                prior |= set(ok["note_csn_id"].astype(str))
            except Exception as e:
                logger.warning(f"could not read prior ids from {pth}: {e}")
        before = len(elig)
        elig = elig[~elig["note_csn_id"].astype(str).isin(prior)]
        logger.info(f"excluded {len(prior)} prior note_csn_ids; eligible {before} -> {len(elig)}")

    if len(elig) < args.n:
        sys.exit(f"Only {len(elig)} eligible notes < requested {args.n}")
    notes_df = elig.sample(n=args.n, random_state=args.seed).reset_index(drop=True)

    done = load_done(args.output)
    logger.info(f"Skipping {len(done)} already-done; sampled {len(notes_df)} notes")
    jobs = [row for _, row in notes_df.iterrows() if str(row["note_csn_id"]) not in done]
    logger.info(f"Reformatting {len(jobs)} notes with {args.model} (repair={'on' if args.repair else 'off'})")

    write_lock = threading.Lock()
    write_header = (not os.path.exists(args.output)) or os.path.getsize(args.output) == 0
    n_ok = 0
    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futs = {ex.submit(process_one, row, "baseline", [], schema, api_key,
                          args.model, None, None, args.repair, args.max_tokens): row
                for row in jobs}
        for fut in as_completed(futs):
            try:
                out = fut.result()
            except Exception as e:
                logger.error(f"note {futs[fut]['note_csn_id']}: {e}")
                continue
            with write_lock:
                pd.DataFrame([out]).reindex(columns=CSV_COLUMNS).to_csv(
                    args.output, mode="a", header=write_header, index=False)
                write_header = False
            if out.get("reformat_success"):
                n_ok += 1
                if n_ok % 25 == 0:
                    logger.info(f"  {n_ok} reformatted OK")
    logger.info(f"Done. {n_ok} newly reformatted -> {args.output}")


if __name__ == "__main__":
    main()
