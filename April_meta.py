import os
from loguru import logger as logging
from pathlib import Path
import torch
import random
import numpy as np
import wandb
import sys

curr_path = os.path.dirname(__file__)
parent_path = os.path.dirname(curr_path)
sys.path.append(parent_path)  # add current terminal path to sys.path
sys.path.append(curr_path)  # add current terminal path to sys.path

# tensorboard --logdir
# tensorboard --logdir runs --host localhost --port 6006


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train_env_file",
        default="meta_scenarios/CVE-2021-3129-train-5.json",
        help="training simulated scenario, i.e. single/env-CVE-2020-2555.json",
    )
    parser.add_argument(
        "--eval_env_file",
        default="meta_scenarios/CVE-2021-3129-eval.json",
        help="evaluating simulated scenario, i.e. single/env-CVE-2020-2555.json",
    )

    parser.add_argument("--mode", default=1, help="running mode (only support Mode 1 and 4 in this project)")

    parser.add_argument("--policy",
                        default="PPO",
                        help="RL training algorithms (only support PPO in this project)")
    parser.add_argument("--meta_algo",
                        default="MAML",
                        help="Meta-RL training algorithm (only support MAML currently)")
    parser.add_argument(
        "--config_file",
        default=None,
        type=str,
        help=
        "config file for RL training algorithms or continual learning method",
    )

    parser.add_argument("--load_agent",
                        default="",
                        type=str,
                        help="load trained agent")
    parser.add_argument("--save_model",
                        action="store_true",
                        default=False,
                        help="support PPO")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--note", type=str, default="", help="wandb note")
    parser.add_argument("--gpu", type=str, default="0", help="gpu id")

    args = parser.parse_args()
    seed = args.seed
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # set seed

    from util import UTIL, Configure, console

    if int(args.mode) in [UTIL.Train_Simulate, UTIL.Eval_Simulate]:

        from Bot import BOT

        UTIL.set_logger()
        UTIL.show_banner()
        
        scenario_path = UTIL.project_path / "scenarios"

        args.train_env_file = scenario_path / args.train_env_file
        if args.eval_env_file:
            args.eval_env_file = scenario_path / args.eval_env_file
        else:
            args.eval_env_file = None
        set_seed(seed)
        use_wandb = Configure.getBool(
            "Train", "use_wandb") if not UTIL.isDebug else False
        use_tensorboard = Configure.getBool("Train", "use_tensorboard")
        Bot = BOT(**vars(args),
                  use_tensorboard=use_tensorboard,
                  use_wandb=use_wandb)
        train_env = Bot.make_env(Bot.train_env_file)
        eval_env = Bot.make_env(
            Bot.eval_env_file) if Bot.eval_env_file else train_env

        if use_wandb:
            UTIL.set_wandb_url()
            Bot.wandb_run = wandb.init(
                project="April-GAP",
                name=Bot.title,
                notes=args.note,
                tags=["your tag"],
                job_type="formal",
                group=
                f"your group",  
                reinit=True,
                allow_val_change=True,
                config=Bot.running_config,
                save_code=False,
            )

        if args.load_agent:
            Bot.load(agent_name=args.load_agent)

        if int(args.mode) == UTIL.Train_Simulate:
            Bot.train(verbose=True, train_env=train_env, eval_env=eval_env)

        elif int(args.mode) == UTIL.Eval_Simulate:
            if not args.load_agent:
                console.print(
                    f"Simulated Evaluation Mode (mode=4) requires loading an trained agent."
                )
                exit(0)
            Bot.Eval_Simulate(env=eval_env, verbose=True, interactive=True)
    else:
        console.print(
            f"Mode error: Only support [bold red]Simulated Training Mode (mode=1)[/] and [bold red]Simulated Evaluation Mode (mode=4)[/]"
        )
        exit(0)
    logging.success("😉  Have a good day~")
    if use_wandb:
        wandb.finish()
