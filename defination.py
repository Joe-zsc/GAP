from prettytable import *
from prettytable.colortable import ColorTable, Themes
from pprint import pprint, pformat
from rich.table import Table
from rich.console import Console
from rich.panel import Panel
from rich import print
from rich.pretty import Pretty
from rich import box
from collections import namedtuple

Attack_Path_Transition = namedtuple(
    "Attack_Path_Transition", ("ip", "step", "action", "result", "reward"))


class Env_data:
    def __init__(
        self,
        ip: str,
        os: str = "",
        port: list = [],
        web_fingerprint: str = "",
        web_fingerprint_component: dict = {},
        services: list = [],
        vulnerability: list = [],
    ):
        self.ip = ip
        self.os = os if os else ""
        self.port = port if port else []
        self.web_fingerprint = web_fingerprint if web_fingerprint else ""
        self.web_fingerprint_component = (web_fingerprint_component
                                          if web_fingerprint_component else {})
        self.services = services if services else []
        self.vulnerability = vulnerability if vulnerability else []

    def dict(self):

        return self.__dict__.copy()


class Host_info(Env_data):
    def __init__(self, ip):
        super().__init__(ip=ip)
        self.session_id: int = -1
        self.session_info: dict = dict()
        self.flag: list = []
        self.flag_path: list = []
        self.neighbor_subnet: list = []
        self.neighbor_host: list = []
        self.intranet_ip: list = []
        self.prior_node = None
        self.pivot: int = 0

    def show_prettytable(self, show=False):
        info = self.__dict__.copy()
        x = ColorTable(theme=Themes.OCEAN)
        x.field_names = ["Name", "Information"]
        for key, value in info.items():
            if value:
                if key == "prior_node":
                    value = value.ip
                if key == "session_id" and value == -1:
                    continue
                x.add_row([key, pformat(value)])
        if show:
            pprint(x)
        return x

    def show_rich_table(self, show=False):

        Targets_table = Table(title="Host Information",
                              highlight=True,
                              box=box.ROUNDED)
        Targets_table.add_column("Name",
                                 justify="left",
                                 style="cyan",
                                 no_wrap=True)

        Targets_table.add_column("Information", style="magenta", no_wrap=True)
        info = self.__dict__.copy()

        for key, value in info.items():
            if value:
                if key == "prior_node":
                    value = value.ip
                if key == "session_id" and value == -1:
                    continue
                Targets_table.add_row(key, pformat(value))
        if show:
            print(Targets_table)
        return Targets_table


# host=Host_info(ip="1111")
# host.port=["22","8080"]
# host.show_prettytable(show=True)
# table=host.show_rich_table(show=True)
# print(Panel(table,box=box.SIMPLE))


class Action_Class:
    use_action_prob = False

    def __init__(
        self,
        id: int,
        name: str,
        act_cost: int,
        success_reward: int = 0,
        type: str = None,
        vulnerability=[],
        exp_info: list[dict] = [],
        description: str = "",
        # setting: dict = None,
    ):
        self.id = id
        self.name = name
        self.description = description
        self.type = type
        self.act_cost = act_cost
        self.success_reward = success_reward

        self.vulnerability = vulnerability
        self.exp_info = exp_info
        # self.setting = setting
        self.set_success_prob()

    def set_success_prob(self):
        """
        Manual
        Great
        Excellent
        Low
        Normal
        Good
        Average
        """
        if (not self.use_action_prob) or (not self.exp_info):
            self.prob = 1
        else:
            rank = self.exp_info["rank"]
            if rank == "Excellent":
                self.prob = 1
            elif rank == "Great":
                self.prob = 0.9
            elif rank == "Good":
                self.prob = 0.7
            elif rank == "Normal":
                self.prob = 0.5
            elif rank == "Average":
                self.prob = 0.5
            elif rank == "Low":
                self.prob = 0.4
            elif rank == "Manual":
                self.prob = 0.5
            else:
                raise ValueError("unknown rank")


# Action_Result = namedtuple(
#     "action_result", ("success", "type", "message", "result_cost")
# )


class Action_Result:
    def __init__(self,
                 success: bool,
                 type: str = "",
                 message="",
                 result_cost: int = 0):
        self.success = success
        self.type = type if type else ""
        self._message = message if message else ""
        self.result_cost = result_cost if result_cost else 0

    def message(self,max_length = 0):
        
        if isinstance(self._message, list):
            message = ", ".join(self._message)
        else:
            message = self._message
        if message:
            result = f"{self.type}: {message}"
            if len(result) > max_length and max_length>0:
                result = result[:max_length] + "..."
            return result
        else:
            return f"{self.type}"


# test=Action_Result(success=True,type="test",message=["22","8080"])
# from rich import print
