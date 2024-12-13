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
from pyecharts.charts import Graph as echartGraph
from pyecharts import options as opts
import IPy
import socket
import random
import subprocess
import nmap
from defination import Host_info
from copy import deepcopy
import ssl
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
 
Well_known_ports = {
    "21": "FTP",
    "22": "SSH",
    "23": "Telnet",
    "25": "SMTP",
    "53": "DNS",
    "67": "DHCP-Server",
    "68": "DHCP-Client",
    "80": "HTTP",
    "110": "POP3",
    "139": "Samba",
    "161": "SNMP",
    "443": "HTTPS",
    "445": "SMB",
    "554": "RTSP",
    "1433": "MSSQL",
    "3306": "MySQL",
    "3389": "RDP",
    "4505": "SaltStack",
    "5432": "PostgreSQL",
}


class IP_:
    def __init__(self, ip, netmask):
        self.address = ip
        self.net_mask = netmask

    def subnet(self):
        return IPy.IP(self.address + "/" + self.net_mask, make_net=True)

    @classmethod
    def checkIP(cls, ip_address):
        try:
            IPy.IP(ip_address)
            return True
        except Exception as e:
            print(str(ip_address) + "不是ip地址,异常原因：" + str(e))
            return False

    @classmethod
    def get_local_ip(cls):
        """
        查询本机ip地址
        :return: ip
        """
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
        except Exception as e:
            # print('attack.get_host_ip '+str(e))
            logging.error("get_local_ip " + str(e))
        finally:
            s.close()
        return ip


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


class Attack_Graph:
    def __init__(self, init):

        self.nodes = []
        self.links = []
        node = opts.GraphNode(name=init.ip,
                              symbol_size=70,
                              category=init.info.pivot)
        self.nodes.append(node)
        self.num = 0

    def addNodes(self, target):

        self.num += 1
        source = target.info.prior_node
        node = opts.GraphNode(name=target.ip,
                              symbol_size=50,
                              category=target.info.pivot)

        self.nodes.append(node)
        edge_label = opts.LabelOpts(
            is_show=True,
            position="middle",
            formatter=str(self.num) + ":" + target.info.vul[0],  # 设置关系说明
        )
        link = opts.GraphLink(source=source.ip,
                              target=target.ip,
                              label_opts=edge_label)
        self.links.append(link)

    def pyecharts_render(self, path="result.html"):

        categories = [
            {
                "name": "Subnet1",
                "itemStyle": {
                    "normal": {
                        "color": "#6950a1",  # 公司颜色为蓝
                        "borderColor": "#c4ccd3",
                        "borderWidth": 1.8,
                    }
                },
            },
            {
                "name": "Subnet2",
                "itemStyle": {
                    "normal": {
                        "color": "#0094f7",  # 公司颜色为蓝
                        "borderColor": "#c4ccd3",
                        "borderWidth": 1.8,
                    }
                },
            },
            {
                "name": "Subnet3",
                "itemStyle": {
                    "normal": {
                        "color": "#f44242",  # 供应商颜色为红
                        "borderColor": "#c4ccd3",
                        "borderWidth": 1.8,
                    }
                },
            },
            {
                "name": "Subnet4",
                "itemStyle": {
                    "normal": {
                        "color": "#b2d235",  # 供应商颜色为红
                        "borderColor": "#c4ccd3",
                        "borderWidth": 1.8,
                    }
                },
            },
        ]

        g = (
            echartGraph().add(
                "",
                self.nodes,
                self.links,
                repulsion=8000,
                categories=categories,
                layout="force",
                is_roam=True,
                is_draggable=True,
                edge_symbol=["circle", "arrow"],
                edge_symbol_size=20,
                # 图的布局。可选：
                # 'none' 不采用任何布局，使用节点中提供的 x， y 作为节点的位置。
                # 'circular' 采用环形布局。
                # 'force' 采用力引导布局。
                is_focusnode=True,
            ).set_global_opts(title_opts=opts.TitleOpts(title="result"))
            #     .render("显示关系说明的关系图.html")
        )
        g.render(path)


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
    local_ip = IP_.get_local_ip()
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
+ -- --=[ Website\t: https://gitee.com/JoeSC/April  ]=-- -- +
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
    def find_file(cls, file_dir: str, suffix: str):
        """
        find all file with spesific suffix, return a dict store their name and path
        e.g.
        dict = { "CVE-2017-0143":"EXP/CVE-2017-0143.py",....}
        """
        all_py_exp_path = {}
        for file in os.listdir(file_dir):
            vul_name = os.path.splitext(file)[0]
            if os.path.splitext(file)[1] == suffix:
                all_py_exp_path[vul_name] = os.path.join(file_dir, file)
        return all_py_exp_path

    @classmethod
    def get_live_hosts(cls, target):
        logging.info("扫描--" + target + "--存活主机")
        nm = nmap.PortScanner()
        nm.scan(hosts=target, arguments="-sP")  # -sn -Pn
        live_hosts = []
        live_hosts = nm.all_hosts()
        if UTIL.local_ip in live_hosts:
            live_hosts.remove(UTIL.local_ip)
        for h in live_hosts:
            logging.info("存活主机有:" + h)
        return live_hosts

    @classmethod
    def check_web_service(cls, host, port):
        # 尝试HTTP
        try:
            with socket.create_connection((host, port), timeout=5) as sock:
                sock.sendall(b"GET / HTTP/1.1\r\nHost: %s\r\n\r\n" %
                             host.encode("ascii"))
                response = sock.recv(1024)
                if b"HTTP/" in response:
                    return "HTTP"

            # 如果不是HTTP，再尝试HTTPS
            with socket.create_connection((host, port), timeout=5) as sock:
                context = ssl.create_default_context()
                with context.wrap_socket(sock, server_hostname=host) as ssock:
                    try:
                        ssock.sendall(b"GET / HTTP/1.1\r\nHost: %s\r\n\r\n" %
                                      host.encode("ascii"))
                        response = ssock.recv(1024)
                        if b"HTTP/" in response:
                            return "HTTPS"
                    except ssl.SSLError:
                        pass  # 忽略SSL错误，因为我们不确定端口是否为HTTPS
            return None
        except (socket.timeout, ConnectionRefusedError, socket.gaierror):
            logging.debug("Connection failed")
            return None
        except Exception as e:
            logging.debug(f"Error: {str(e)}")
            return None

    @classmethod
    def generate_lport(cls):
        cls.lport = cls.lport + 1
        while cls.lport in cls.lport_list:
            cls.lport += 1
        cls.lport_list.append(cls.lport)
        return cls.lport

    @classmethod
    def set_reverse_ip(cls, target_info: Host_info):
        reverse_ip: str = "0.0.0.0"
        prior_node = target_info.prior_node
        if not prior_node:  # 如果没有父节点，默认为本地
            reverse_ip = cls.local_ip
        else:
            prior_node_info: Host_info = prior_node.info
            if not prior_node_info.intranet_ip:  # 如果父节点没有内网地址
                reverse_ip = prior_node_info.ip
            else:
                intranet_ip = prior_node_info.intranet_ip
                target_ip = target_info.ip
                for i_ip in intranet_ip:
                    if target_ip in i_ip.subnet():
                        reverse_ip = i_ip.address
                        break

        logging.info(f"set reverse ip {reverse_ip}")
        return reverse_ip

    @classmethod
    def exec_shell_command(cls, command, root=False):
        password = UTIL.password
        # status代表返回状态，0表示成功
        # result代表命令的返回结果
        if root:
            (status, result) = subprocess.getstatusoutput(
                "echo %s| sudo -S %s" % (password, command))
        else:
            (status, result) = subprocess.getstatusoutput(command)
        if not status:
            logging.debug(f"run {command} success : \n{result}")
        else:
            logging.debug(f"run {command} failed:\n{result}")
        return status, result

    # @classmethod
    # def add_proxy(cls, ip, port):
    #     file = "/etc/proxychains.conf"
    #     s = "socks4 " + ip + ' ' + port + '/n'
    #     with open(file, 'a+') as f:
    #         f.write(s)

    

    @classmethod
    def set_logger(cls, print_lever="INFO", logfile_level="DEBUG"):
        """
        logger.debug("详细调试信息")
        logger.info("普通信息")
        logger.success("成功信息")
        logger.warning("警告信息")
        logger.error("错误信息")
        logger.trace("异常信息")
        logger.critical("严重错误信息")
        """
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
