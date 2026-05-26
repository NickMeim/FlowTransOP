from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from .backends import RuntimeBackends

AUTOTRANSOP_NOTE = """
IMPORTANT AutoTransOP note:
AutoTransOP/CPA-style baselines are highly sensitive to hyperparameters. The
choice of mutual-information losses, cosine and/or Euclidean latent penalties,
and prior/adversarial discriminators from the original publication is a modeling
decision that must be re-tuned for the user's data, paired-sample regime, and
feature space. Do not treat the checked-in defaults as universally optimal.
"""

L1000_METHOD_SCRIPTS = {
    "flowtransop": "AutoTransOP_Pretrain_FlowMatch.py",
    "consensus-decoders": "DecodeFromConsencusSpace.py",
    "consensus-decoders-different-inputs": "DecodeFromConsencusSpace_diffenetInputs.py",
    "consensus-decoders-bracketed": "DecodeFromConsencusSpace_diffenetInputs_bracketed.py",
    "hybrid-flowtransop": "FlowMatch_lowPairsPercentage_PairsAndSimilarity.py",
    "hybrid-flowtransop-extreme": "FlowMatch_lowPairsPercentageExtreme_PairsAndSimilarity.py",
    "hybrid-flowtransop-extreme-mean": "FlowMatch_lowPairsPercentageExtreme_PairsAndSimilarity_meanAgg.py",
    "hybrid-flowtransop-extreme-sum": "FlowMatch_lowPairsPercentageExtreme_PairsAndSimilarity_sumAgg.py",
    "autotransop": "AutoTransOP_lowPairsPercentageExtreme.py",
    "simple-autotransop": "AutoTransOP_lowPairsPercentageExtreme.py",
}


def _repo_root(path: str | None) -> Path:
    return Path(path).resolve() if path else Path.cwd().resolve()


def _run_script(
    repo_root: Path,
    script: str,
    passthrough: list[str],
    env_overrides: dict[str, str] | None = None,
) -> int:
    script_path = repo_root / "learning" / script
    if not script_path.exists():
        raise FileNotFoundError(f"Cannot find script: {script_path}")
    cmd = [sys.executable, str(script_path), *passthrough]
    env = os.environ.copy()
    if env_overrides:
        env.update(env_overrides)
    return subprocess.call(cmd, cwd=repo_root / "learning", env=env)


def _add_backend_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model-device",
        default="cuda",
        help="Torch device for model training/evaluation wrappers. Default: cuda.",
    )
    parser.add_argument(
        "--transact-backend",
        choices=["gpu", "cpu"],
        default="gpu",
        help="TRANSACT/pre-alignment backend to expose to package workflows. Default: gpu.",
    )
    parser.add_argument(
        "--transact-device",
        default="cuda",
        help="Device for the TRANSACT/pre-alignment backend. Default: cuda.",
    )


def _backend_env(args: argparse.Namespace) -> dict[str, str]:
    return RuntimeBackends(
        model_device=args.model_device,
        transact_backend=args.transact_backend,
        transact_device=args.transact_device,
    ).as_env()


def _print_autotransop_note() -> None:
    print(AUTOTRANSOP_NOTE.strip(), file=sys.stderr, flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="flowtransop")
    sub = parser.add_subparsers(dest="command", required=True)

    predict = sub.add_parser("predict", help="Translate a preprocessed numpy matrix with a saved checkpoint.")
    predict.add_argument("--normal-checkpoint", required=True)
    predict.add_argument("--m2h-checkpoint", default=None)
    predict.add_argument("--direction", required=True, choices=["h2m", "human-to-mouse", "m2h", "mouse-to-human"])
    predict.add_argument("--input-npy", required=True)
    predict.add_argument("--output-npy", required=True)
    predict.add_argument("--model-device", "--device", dest="model_device", default=None)
    predict.add_argument("--batch-size", type=int, default=256)
    predict.add_argument("--n-steps", type=int, default=10)

    train_cv = sub.add_parser(
        "train-archs4-fold",
        help="Run the ARCHS4 CV training scripts.",
        epilog="Unrecognized arguments are passed through to the underlying learning script.",
    )
    train_cv.add_argument("--repo-root", default=None, help="Repository root. Defaults to the current directory.")
    train_cv.add_argument("--fold", required=True)
    train_cv.add_argument("--direction", choices=["h2m", "m2h"], default="h2m")
    _add_backend_args(train_cv)

    train_full = sub.add_parser(
        "train-archs4-ensemble",
        help="Run one full-data ARCHS4 ensemble member.",
        epilog="Unrecognized arguments are passed through to train_ARCHS4_full_ensemble.py.",
    )
    train_full.add_argument("--repo-root", default=None, help="Repository root. Defaults to the current directory.")
    train_full.add_argument("--ensemble-id", default="0")
    train_full.add_argument("--fold", default="0")
    _add_backend_args(train_full)

    evaluate = sub.add_parser(
        "evaluate-archs4-fold",
        help="Run latent/cycle/orthologue/expression/liver evaluations for one fold.",
        epilog="Unrecognized arguments are passed through to evaluate_translation.py.",
    )
    evaluate.add_argument("--repo-root", default=None, help="Repository root. Defaults to the current directory.")
    evaluate.add_argument("--fold", required=True)
    evaluate.add_argument("--include-liver", action="store_true")
    _add_backend_args(evaluate)

    score = sub.add_parser(
        "score-mash",
        help="Run the final MASH PLSR scoring workflow.",
        epilog="Unrecognized arguments are passed through to score_liver_mas_fibrosis_final_expression_mean.py.",
    )
    score.add_argument("--repo-root", default=None, help="Repository root. Defaults to the current directory.")
    _add_backend_args(score)

    l1000 = sub.add_parser(
        "run-l1000",
        help="Run one L1000 benchmark workflow from the learning scripts.",
        epilog="Unrecognized arguments are passed through to the selected learning script.",
    )
    l1000.add_argument("--repo-root", default=None, help="Repository root. Defaults to the current directory.")
    l1000.add_argument(
        "--method",
        required=True,
        choices=sorted(L1000_METHOD_SCRIPTS),
        help="L1000 workflow to run.",
    )
    _add_backend_args(l1000)

    args, extra = parser.parse_known_args(argv)
    root = _repo_root(getattr(args, "repo_root", None))

    if args.command == "predict":
        if extra:
            parser.error(f"predict does not accept extra arguments: {' '.join(extra)}")
        from .inference import translate_array

        out = translate_array(
            input_npy=args.input_npy,
            output_npy=args.output_npy,
            normal_checkpoint=args.normal_checkpoint,
            m2h_checkpoint=args.m2h_checkpoint,
            direction=args.direction,
            device=args.model_device,
            batch_size=args.batch_size,
            n_steps=args.n_steps,
        )
        print(out)
        return 0

    if args.command == "train-archs4-fold":
        script = "train_ARCHS4_fold.py" if args.direction == "h2m" else "train_ARCHS4_fold_m2h.py"
        return _run_script(root, script, ["--fold", str(args.fold), *extra], _backend_env(args))

    if args.command == "train-archs4-ensemble":
        return _run_script(
            root,
            "train_ARCHS4_full_ensemble.py",
            ["--ensemble_id", str(args.ensemble_id), "--fold", str(args.fold), *extra],
            _backend_env(args),
        )

    if args.command == "evaluate-archs4-fold":
        env = _backend_env(args)
        code = _run_script(root, "evaluate_translation.py", ["--fold", str(args.fold), *extra], env)
        if code != 0:
            return code
        code = _run_script(root, "evaluate_expression_mmd_archs4.py", ["--fold", str(args.fold)], env)
        if code != 0 or not args.include_liver:
            return code
        return _run_script(root, "evaluate_liver.py", ["--fold", str(args.fold)], env)

    if args.command == "score-mash":
        return _run_script(root, "score_liver_mas_fibrosis_final_expression_mean.py", extra, _backend_env(args))

    if args.command == "run-l1000":
        script = L1000_METHOD_SCRIPTS[args.method]
        if args.method in {"autotransop", "simple-autotransop"}:
            _print_autotransop_note()
        return _run_script(root, script, extra, _backend_env(args))

    parser.error(f"Unhandled command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
