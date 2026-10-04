"""
Command-line entry point for the biometrics package. One user, one model: all
locations are fixed in biometrics.paths.

    biometrics collect         -- record labelled keystroke sessions into data/raw
    biometrics process         -- turn data/raw into the processed training dataset
    biometrics train           -- train the model in models/xgboost
    biometrics infer           -- live inference + the protection agent (optionally reporting to a server)
    biometrics refine          -- learn the sessions flagged for review (re-authenticated alerts) and retrain
    biometrics describe-model  -- regenerate the agent's model artifacts without retraining
"""
import argparse
import datetime
import time
from sys import exit

from biometrics import paths


def cmd_collect(args):
    from biometrics.collector.keylogger import KeyLogger

    label = args.label or input("Enter label name: ")
    input("Press enter to start logging (Esc to cancel)")
    logger = KeyLogger(label, str(paths.RAW_DIR), threshold=args.threshold)

    time.sleep(0.5)  # let the Enter keypress clear before we start capturing

    try:
        logger.start_logging()
    except KeyboardInterrupt:
        logger.end_logging()
    finally:
        paths.hand_to_user(paths.RAW_DIR)
    exit(0)


def cmd_process(args):
    from biometrics.processing.pipeline import process

    process(str(paths.RAW_DIR), str(paths.PROCESSED_CSV))
    paths.hand_to_user(paths.PROCESSED_CSV.parent)


def cmd_train(args):
    from biometrics.training.train_xgboost import train

    train(str(paths.PROCESSED_CSV), str(paths.MODEL_DIR), n_estimators=args.n_estimators, use_gpu=args.gpu)
    paths.hand_to_user(paths.MODEL_DIR)


def cmd_infer(args):
    from biometrics.agent.agent import Agent
    from biometrics.collector.keylogger import KeyLogger

    agent = Agent(host_id=args.host_id, server_url=args.server)
    agent.start()
    where = f"reporting to {args.server}" if args.server else "no dashboard server (local only)"
    print(f"Protection agent running as {agent.host_id}, {where}")

    label = args.label or datetime.datetime.now().strftime("session_%Y-%m-%d_%H-%M-%S")
    input("Press enter to start inference (Esc to cancel)")
    logger = KeyLogger(label, str(paths.INFER_OUTPUT_DIR), threshold=args.threshold,
                       model_paths=paths.model_paths(), on_prediction=agent.on_prediction)
    agent.on_model_reloaded.append(logger.reload_model)

    time.sleep(0.5)

    try:
        logger.start_logging()
    except KeyboardInterrupt:
        logger.end_logging()
    finally:
        agent.stop()
    exit(0)


def cmd_refine(args):
    from biometrics.training.review_queue import pending, refine

    sessions = pending()
    if not sessions:
        print("Nothing flagged for review. Sessions appear here after you re-authenticate "
              "at a challenge or an alert.")
        return

    print(f"{len(sessions)} session(s) flagged for review. The model thought they were someone else, "
          f"and you re-authenticated:")
    for s in sessions:
        score = f"{s['impostor_score']:.0%}" if s["impostor_score"] is not None else "?"
        print(f"  {s['name']}  {s['keystrokes']} keystrokes, {s['level'] or 'unknown level'}, impostor score {score}")

    if not args.yes and input("Learn all of them as you (Main user) and retrain? [y/N]: ").strip().lower() \
            not in ("y", "yes"):
        print("Cancelled; nothing changed.")
        return

    result = refine()
    print(f"Done: +{result['sessions_added']} sessions / {result['keystrokes_added']} keystrokes, "
          f"accuracy {result['accuracy']:.2%}. A running agent picks the new model up after restart, "
          f"or refine from the dashboard to switch it live.")


def cmd_describe_model(args):
    from biometrics.training.train_xgboost import describe_model

    result = describe_model(str(paths.PROCESSED_CSV), str(paths.MODEL_DIR))
    paths.hand_to_user(paths.MODEL_DIR)
    t = result["thresholds"]
    print(f"Accuracy {result['accuracy']:.4f}, tau_warn={t['tau_warn']:.4f}, "
          f"tau_critical={t['tau_critical']:.4f}, EER={t['eer']:.4f}, AUC={t['auc']:.4f}")
    print(f"Wrote thresholds.json, model_info.json, feature_profile.json to {paths.MODEL_DIR}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="biometrics", description="Keystroke biometric identification toolkit")
    subparsers = parser.add_subparsers(dest="command", required=True)

    collect = subparsers.add_parser("collect", help="Record labelled keystroke sessions into data/raw")
    collect.add_argument("--label", help="Label for this session: 'User' for the owner (prompted if omitted)")
    collect.add_argument("--threshold", type=int, default=50, help="Keystrokes per saved session file")
    collect.set_defaults(func=cmd_collect)

    process = subparsers.add_parser("process", help="Turn data/raw into the processed training dataset")
    process.set_defaults(func=cmd_process)

    train = subparsers.add_parser("train", help="Train the model")
    train.add_argument("--n-estimators", type=int, default=100)
    train.add_argument("--gpu", action="store_true", help="Train using CUDA if available")
    train.set_defaults(func=cmd_train)

    infer = subparsers.add_parser("infer", help="Live inference + the protection agent")
    infer.add_argument("--label", help="Session label")
    infer.add_argument("--threshold", type=int, default=15, help="Keystrokes per prediction window")
    infer.add_argument("--server", help="Dashboard server URL to report to, e.g. http://127.0.0.1:8642 "
                                         "(omit to run the agent locally without reporting)")
    infer.add_argument("--host-id", help="Name this machine is shown under on the dashboard (default: hostname)")
    infer.set_defaults(func=cmd_infer)

    refine = subparsers.add_parser("refine", help="Learn the sessions flagged for review and retrain")
    refine.add_argument("-y", "--yes", action="store_true", help="Don't ask for confirmation")
    refine.set_defaults(func=cmd_refine)

    describe = subparsers.add_parser(
        "describe-model", help="Regenerate thresholds/model_info/feature_profile for the trained model")
    describe.set_defaults(func=cmd_describe_model)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
