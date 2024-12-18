import configparser
from loguru import logger as logging
import time
import json
import yaml
from pprint import pprint, pformat
import pandas as pd
from pathlib import Path
from datetime import datetime
import sys, os
from easydict import EasyDict
from colorama import init, Fore, Back, Style
import time
import random
from copy import deepcopy
from rich.console import Console
import wandb
import torch
import numpy as np
curr_path = os.path.dirname(__file__)
parent_path = os.path.dirname(curr_path)
sys.path.append(parent_path)  # add current terminal path to sys.path
sys.path.append(curr_path)  # add current terminal path to sys.path

console = Console()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
 

class Configure:
    conf = configparser.ConfigParser()
    try:
        result = conf.read(Path(__file__).parent / "config.ini")
    except Exception as e:
        logging.error("config file not found" + e)

    @classmethod
    def get(cls, label, name):
        return cls.conf.get(label, name)

    @classmethod
    def read_configure_value(cls, sections=[]):
        config_dict = {}
        for section in cls.conf.sections():
            if not sections or section in sections:
                config_dict[section] = {}
                # 遍历section中的每个option
                for option in cls.conf.options(section):
                    # 将option和对应的值添加到子字典中
                    config_dict[section][option] = cls.conf.get(
                        section, option)
        return config_dict

    @classmethod
    def getBool(cls, label, name):
        bl = cls.conf.get(label, name)
        if bl.lower() == "true" or bl == "1":
            return True
        elif bl.lower() == "false" or bl == "0":
            return False
        else:
            raise ValueError("Bool value must be true/1/True/false/False/0")

    @classmethod
    def set(cls, label, name, value):
        cls.conf.set(label, name, str(value))
        cls.conf.write(open("config.ini", "w"))


class Metric:
    Train_Metric = EasyDict({"all": [], "best": {}})
    """
    Training signals
    """
    EarlyTerminate = "early_terminate"
    Finished = "finished"
    Failed = "failed"
    Success = "success"

    def __init__(self) -> None:
        pass

class UTIL:
    """
    Running Mode:
    """

    isDebug = True if sys.gettrace() else False
    Manual = 0
    # Train_Manual = 1
    Train_Simulate = 1
    # Train_Auto = 2
    Train_Real = 2
    # Real_Attack = 3
    Eval_Real = 3
    # Evaluation = 4
    Eval_Simulate = 4

    today = datetime.now().strftime("%b%d")
    current_time = datetime.now().strftime("%b%d_%H-%M-%S")
    lport = random.randint(10000, 20000)
    lport_list = []
    Running_title = ""
    project_name = Path(__file__).parent.stem

    project_path = Path(__file__).parent
    scenario_path = project_path / "scenarios"
    trained_agent_path = project_path / Configure.get("Train",
                                                      "trained_agent_path")
    log_path = project_path / "log"
    running_record_path = project_path / "running_record"

    password = ""

    neighbor_discovery = True

    def __init__(self) -> None:
        pass

    @classmethod
    def mode_name(cls, mode):
        if mode == 0:
            return "Manual Mode"
        if mode == 1:
            return "Simulated Training Mode"
        if mode == 2:
            return "Real Training Mode"
        if mode == 3:
            return "Simulated Evaluation Mode"
        if mode == 4:
            return "Real Evaluation Mode"

    @classmethod
    def show_banner(cls):

        banner = """
 
     ___      .______   .______       __   __      
    /   \     |   _  \  |   _  \     |  | |  |     
   /  ^  \    |  |_)  | |  |_)  |    |  | |  |     
  /  /_\  \   |   ___/  |      /     |  | |  |     
 /  _____  \  |  |      |  |\  \----.|  | |  `----.
/__/     \__\ | _|      | _| `._____||__| |_______|
                                                   

"""

        print(banner)
        cls.show_credit()
        time.sleep(2)

    # flag_log
    @classmethod
    def show_credit(cls):
        credit = """
+ -- --=[ APRIL\t: Autonomous Penetesting based on ReInforcement Learning             ]=-- -- +
+ -- --=[ Author\t: NUDT-HFBOT Team                                   ]=-- -- +
+ -- --=[ Website\t: https://github.com/Joe-zsc/GAP  ]=-- -- +
    """
        print(credit)

    @classmethod
    def line_break(cls, length=60, symbol="-"):
        line_break = symbol * length
        logging.info(line_break)

    def write_csv_DictList(file: Path, data: list):
        variables = list(data[0].keys())
        pd_data = pd.DataFrame([[i[j] for j in variables] for i in data],
                               columns=variables)
        pd_data.to_csv(file, mode="w", index=False)
        return pd_data

    @classmethod
    def smooth_data(cls, data: list, weight: float = 0.9):
        smoothed_data = []

        last = data[0]
        smoothed = []
        for point in data:
            smoothed_val = last * weight + (1 - weight) * point
            smoothed_data.append(smoothed_val)
            last = smoothed_val
        return smoothed_data

    @classmethod
    def write_to_csv(cls, data: list, save_path: str):
        """
        data: [dict1,dict2,...]
        """
        import csv

        assert len(data) > 0, "the input data is empty"

        f = open(save_path, "a", encoding="utf8", newline="")
        writer = csv.DictWriter(f, fieldnames=data[0].keys())
        writer.writeheader()
        for line in data:
            writer.writerow(line)

    @classmethod
    def save_json(cls, path, data):
        with open(path, "w", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False, indent=4))

    @classmethod
    def read_yaml(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            result = yaml.load(f.read(), Loader=yaml.FullLoader)
        return result

    @classmethod
    def set_logger(cls, print_lever="INFO", logfile_level="DEBUG"):
       
        log_file = cls.project_path / "log" / f"{Path(__file__).parent.stem}.log"

        logging.remove()
        logging.add(sys.stderr,
                    level=print_lever,
                    backtrace=True,
                    diagnose=True)
        logging.add(
            log_file,
            level=logfile_level,
            encoding="UTF-8",
            rotation="1 day",
            colorize=True,
            backtrace=True,
            diagnose=True,
            format=
            "{time:YYYY-MM-DD HH:mm:ss} - {level} - {file} - {line} - {message}",
        )
        logging.level("NOTE", no=35, color="<cyan>",icon='✨')

    @classmethod
    def set_wandb_url(cls):
        try:
            os.environ["WANDB_BASE_URL"] = Configure.get(
                "wandb", "WANDB_BASE_UR")
            wandb.login(key=Configure.get("wandb", "API_Key"))
        except:
            os.environ["WANDB_BASE_URL"] = "https://api.wandb.ai"


def split_num_l(num_lst):
    """merge successive num, sort lst(ascending or descending): 'as' or 'des'
    eg: [1, 3,4,5,6, 9,10] -> [[1], [3, 4, 5, 6], [9, 10]]
    """
    num_lst_tmp = [int(n) for n in num_lst]
    sort_lst = sorted(num_lst_tmp)  # ascending
    len_lst = len(sort_lst)
    i = 0
    split_lst = []

    tmp_lst = [sort_lst[i]]
    while True:
        if i + 1 == len_lst:
            break
        next_n = sort_lst[i + 1]
        if sort_lst[i] + 1 == next_n:
            tmp_lst.append(next_n)
        else:
            split_lst.append(tmp_lst)
            tmp_lst = [next_n]
        i += 1
    split_lst.append(tmp_lst)
    return split_lst


def Merge_str_lst(num_lst):
    """[[1], [3, 4, 5, 6], [9, 10]] -> ['1', '3~6', '9~10']"""
    if not num_lst:
        return []
    mylst = split_num_l(num_lst)
    mg_l = []
    for num_l in mylst:
        if len(num_l) == 1:
            mg_l.append(str(num_l[0]))
        else:
            mg_l.append(str(num_l[0]) + "~" + str(num_l[-1]))
    return mg_l


class color:

    PURPLE = "\033[95m"
    CYAN = "\033[96m"
    DARKCYAN = "\033[36m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    GREY = "\033[1;30m"
    END = "\033[0m"
    init(autoreset=True)

    @classmethod
    def print(cls, s, c=GREEN, end="\n"):
        print(c + s + cls.END, end=end)

    @classmethod
    def color_str(cls, s, c=GREEN):
        s = pformat(s)
        return c + s + cls.END

    #  前景色:红色  背景色:默认
    @classmethod
    def red(cls, s):
        return Fore.RED + s + Fore.RESET

    #  前景色:绿色  背景色:默认
    @classmethod
    def green(cls, s):
        return Fore.GREEN + s + Fore.RESET

    #  前景色:黄色  背景色:默认
    @classmethod
    def yellow(cls, s):
        return Fore.YELLOW + s + Fore.RESET

    #  前景色:蓝色  背景色:默认
    @classmethod
    def blue(cls, s):
        return Fore.BLUE + s + Fore.RESET

    #  前景色:洋红色  背景色:默认
    @classmethod
    def magenta(cls, s):

        return Fore.MAGENTA + s + Fore.RESET

    #  前景色:青色  背景色:默认
    @classmethod
    def cyan(cls, s):
        return Fore.CYAN + s + Fore.RESET

    #  前景色:白色  背景色:默认
    @classmethod
    def white(cls, s):
        return Fore.WHITE + s + Fore.RESET

    #  前景色:黑色  背景色:默认
    @classmethod
    def black(cls, s):
        return Fore.BLACK

    #  前景色:白色  背景色:绿色
    @classmethod
    def white_green(cls, s):
        return Fore.WHITE + Back.GREEN + s

    @classmethod
    def dave(cls, s):
        return Style.BRIGHT + Fore.GREEN + s




Configure.read_configure_value(sections=["Embedding", "Support", "Exploit"])
