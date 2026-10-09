"""Command-line interface.

Examples
--------
    python -m tsr build                 # download gallery + build index
    python -m tsr download --max-images 10000
    python -m tsr build-index
    python -m tsr make-eval --num-queries 300
    python -m tsr eval
    python -m tsr serve --port 8000
    python -m tsr stats
"""

from __future__ import annotations

import argparse
import sys

from tsr.config import load_config
from tsr.logging_utils import get_logger, setup_logging

logger = get_logger(__name__)


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default=None, help="Path to a YAML config file")
    parser.add_argument("--log-level", default="INFO", help="Logging level (default INFO)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tsr", description="Text-to-Image Retrieval (OpenCLIP)")
    sub = parser.add_subparsers(dest="command", required=True)

    p_download = sub.add_parser("download", help="Download/collect images and write the manifest")
    _add_common(p_download)
    p_download.add_argument("--source", default=None, choices=["coco", "coco_hf", "local"])
    p_download.add_argument("--max-images", type=int, default=None)

    p_index = sub.add_parser("build-index", help="Embed images and build the FAISS index")
    _add_common(p_index)

    p_build = sub.add_parser("build", help="download + build-index")
    _add_common(p_build)
    p_build.add_argument("--source", default=None, choices=["coco", "coco_hf", "local"])
    p_build.add_argument("--max-images", type=int, default=None)

    p_eval = sub.add_parser("eval", help="Evaluate retrieval on the eval set")
    _add_common(p_eval)
    p_eval.add_argument("--queries", default=None, help="Path to a queries JSON file")
    p_eval.add_argument("--hybrid", action="store_true", default=None, help="Enable hybrid search")
    p_eval.add_argument("--top-k", type=int, nargs="*", default=None)

    p_mkeval = sub.add_parser("make-eval", help="Generate eval queries from gallery captions")
    _add_common(p_mkeval)
    p_mkeval.add_argument("--num-queries", type=int, default=None)

    p_serve = sub.add_parser("serve", help="Run the web server")
    _add_common(p_serve)
    p_serve.add_argument("--host", default=None)
    p_serve.add_argument("--port", type=int, default=None)
    p_serve.add_argument("--reload", action="store_true")

    p_stats = sub.add_parser("stats", help="Print gallery/index statistics")
    _add_common(p_stats)

    p_thumbs = sub.add_parser("thumbnails", help="Pre-generate cached thumbnails")
    _add_common(p_thumbs)
    p_thumbs.add_argument("--size", type=int, default=480)

    p_clean = sub.add_parser("clean", help="Delete caches (optionally all generated data)")
    _add_common(p_clean)
    p_clean.add_argument("--all", action="store_true", help="Also delete gallery/index/db/manifest")
    p_clean.add_argument("--yes", action="store_true", help="Skip confirmation prompt")

    p_onnx = sub.add_parser("export-onnx", help="Export the CLIP model to ONNX (fast CPU)")
    _add_common(p_onnx)
    p_onnx.add_argument("--out", default=None, help="Output directory (default: model.onnx_dir)")

    p_caption = sub.add_parser("caption", help="Generate captions for local images (BLIP)")
    _add_common(p_caption)
    p_caption.add_argument("--root", default=None, help="Image folder (default: gallery.local_root)")
    p_caption.add_argument("--out", default=None, help="Output captions.json (default: <root>/captions.json)")
    p_caption.add_argument("--model", default=None, help="Captioning model override")

    p_compare = sub.add_parser("compare", help="Compare search configs with bootstrap CIs")
    _add_common(p_compare)
    p_compare.add_argument("--queries", default=None, help="Path to a queries JSON file")
    p_compare.add_argument("--boot", type=int, default=1000, help="Bootstrap resamples")

    return parser


def _load(args) -> object:
    cfg = load_config(args.config)
    setup_logging(args.log_level)
    return cfg


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg = _load(args)

    if args.command == "download":
        from tsr.gallery import prepare_gallery

        if args.source:
            cfg.gallery["source"] = args.source
        count = prepare_gallery(cfg, max_images=args.max_images)
        logger.info("Done. %d images in the gallery.", count)
        return 0

    if args.command == "build-index":
        from tsr.pipeline import build_index

        size = build_index(cfg)
        logger.info("Done. Index has %d vectors.", size)
        return 0

    if args.command == "build":
        from tsr.gallery import prepare_gallery
        from tsr.pipeline import build_index, make_eval_set, pregenerate_thumbnails

        if args.source:
            cfg.gallery["source"] = args.source
        count = prepare_gallery(cfg, max_images=args.max_images)
        logger.info("Prepared %d images.", count)
        size = build_index(cfg)
        make_eval_set(cfg)
        if bool(cfg.gallery.get("pregenerate_thumbnails", False)):
            pregenerate_thumbnails(cfg)
        logger.info("Build complete. Index has %d vectors.", size)
        return 0

    if args.command == "make-eval":
        from tsr.pipeline import make_eval_set

        make_eval_set(cfg, num_queries=args.num_queries)
        return 0

    if args.command == "eval":
        from tsr.pipeline import run_evaluation

        if args.hybrid:
            cfg.search["hybrid"] = True
        if args.top_k:
            cfg.eval["top_k"] = args.top_k
        run_evaluation(cfg, queries_file=args.queries)
        return 0

    if args.command == "serve":
        import uvicorn

        host = args.host or cfg.app.host
        port = args.port or int(cfg.app.port)
        logger.info("Serving on http://%s:%s", host, port)
        uvicorn.run(
            "tsr.api:app",
            host=host,
            port=int(port),
            reload=bool(args.reload),
            factory=False,
        )
        return 0

    if args.command == "stats":
        from tsr.config import db_path, index_path, manifest_path

        mp = manifest_path(cfg)
        ip = index_path(cfg)
        dp = db_path(cfg)
        logger.info("data dir : %s", cfg.paths.data_dir)
        logger.info("manifest : %s (%s)", mp, "exists" if mp.exists() else "MISSING")
        logger.info("index    : %s (%s)", ip, "exists" if ip.exists() else "MISSING")
        logger.info("database : %s (%s)", dp, "exists" if dp.exists() else "MISSING")
        if dp.exists():
            from tsr.db import GalleryDB

            db = GalleryDB(dp)
            logger.info("images   : %d", db.count())
            db.close()
        return 0

    if args.command == "thumbnails":
        from tsr.pipeline import pregenerate_thumbnails

        pregenerate_thumbnails(cfg, size=args.size)
        return 0

    if args.command == "export-onnx":
        from tsr.onnx_export import export_clip_to_onnx

        export_clip_to_onnx(cfg, out_dir=args.out)
        return 0

    if args.command == "caption":
        from tsr.captioning import caption_local_images

        caption_local_images(cfg, root=args.root, out=args.out, model=args.model)
        return 0

    if args.command == "compare":
        from tsr.pipeline import compare_search_configs

        compare_search_configs(cfg, queries_file=args.queries, n_boot=args.boot)
        return 0

    if args.command == "clean":
        import shutil

        from tsr.config import (
            cache_dir,
            data_dir,
            db_path,
            eval_queries_path,
            ids_path,
            images_dir,
            index_path,
            manifest_path,
        )

        targets = [cache_dir(cfg)]
        if args.all:
            targets += [
                images_dir(cfg),
                index_path(cfg),
                ids_path(cfg),
                db_path(cfg),
                manifest_path(cfg),
                eval_queries_path(cfg),
            ]
        if args.all and not args.yes:
            reply = input(f"Delete {len(targets)} paths under {data_dir(cfg)}? [y/N] ")
            if reply.strip().lower() not in {"y", "yes"}:
                logger.info("Aborted.")
                return 1
        removed = 0
        for target in targets:
            if target.is_dir():
                shutil.rmtree(target, ignore_errors=True)
                removed += 1
            elif target.exists():
                target.unlink()
                removed += 1
        logger.info("Removed %d path(s).", removed)
        return 0

    parser.error(f"unknown command {args.command!r}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
