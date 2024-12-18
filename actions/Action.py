
from loguru import logger as logging
import json
import sys
import os
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
curr_path = os.path.dirname(__file__)
parent_path = os.path.dirname(curr_path)
sys.path.append(parent_path)  # add current terminal path to sys.path
sys.path.append(curr_path)  # add current terminal path to sys.path
# from NLP_Module.sentence_vector import get_vector
from util import Configure, UTIL
from defination import Host_info, Action_Class, Action_Result
from NLP_Module.Encoder import *



class Action:
    action_vulnerabilities = (
        UTIL.project_path / "actions" /
        Configure.get("common", "action_vulnerabilities_file"))

    with open(action_vulnerabilities / "actions.json", "r",
              encoding="UTF-8") as f:  # *********
        vulhub = json.loads(f.read())
    Vul_set = vulhub["actions"]

    Vul_cve_set = []

    for v in Vul_set:
        Vul_cve_set += v["vulnerability"]

    Vul_cve_set = list(set(Vul_cve_set))

    All_EXP = [
        Action_Class(
            id=int(v["id"]),
            name=v["name"],
            act_cost=10,  # 10
            success_reward=1000,
            type=v["type"],
            description=v["description"],
            vulnerability=v["vulnerability"],
            exp_info=v["exp_info"] if "exp_info" in v.keys() else "",
            # setting=v["setting"] if "setting" in v.keys() else "",
        ) for v in Vul_set
    ]
    assert len(All_EXP) == len(
        set(All_EXP)), f"{All_EXP-set(All_EXP)}"  
    PORT_SCAN = Action_Class(id=-1,
                             name="Port Scan",
                             act_cost=0,
                             success_reward=0,
                             type="Scan")
    OS_SCAN = Action_Class(id=-2,
                           name="OS Detect",
                           success_reward=100,
                           act_cost=0,
                           type="Scan")
    SERVICE_SCAN = Action_Class(id=-3,
                                name="Service Scan",
                                success_reward=100,
                                act_cost=0,
                                type="Scan")
    PORT_SERVICE_SCAN = Action_Class(id=-4,
                                     name="Port&Service Scan",
                                     act_cost=0,
                                     type="Scan")
    WEB_SCAN = Action_Class(id=-5,
                            name="Web Scan",
                            act_cost=0,
                            success_reward=100,
                            type="Scan")
    Scan_actions = [PORT_SCAN, SERVICE_SCAN, OS_SCAN, WEB_SCAN]
    legal_actions = Scan_actions + All_EXP
    legal_actions_name = [action.name for action in legal_actions]
    action_space = len(legal_actions)

    action_repetition = Action_Result(success=False,
                                      type="action repetition",
                                      message="action_repetition",
                                      result_cost=5)
    PORT_required = Action_Result(
        success=False,
        type="Execution conditions not met",
        message="Require port information",
        result_cost=10,
    )
    SERVICE_required = Action_Result(
        success=False,
        type="Execution conditions not met",
        message="Require services information",
        result_cost=10,
    )
    OS_required = Action_Result(
        success=False,
        type="Execution conditions not met",
        message="Require OS information",
        result_cost=10,
    )
    WebFingerprint_required = Action_Result(
        success=False,
        type="Execution conditions not met",
        message="Require WebFingerprint information",
        result_cost=10,
    )
    real_exploit_times = 0
    
    def __init__(self):
        self.history_set = set()
        self.ExpExecuteCount = (
            self.count_ExpAction()
        )  # Train_Real mode use it to speed up training process
        self.last_action_id = -999
        self.webscan_times = 0
        self.exp_coverage = 0
    def reset(self):
        self.history_set = set()

    def count_ExpAction(self):
        vuls = []
        for v in self.All_EXP:
            vuls.append(v.name)
        count = dict.fromkeys(vuls, 0)
        return count

    def count_ExpCoverage(self):
        count = sum(value > 0 for value in self.ExpExecuteCount.values())
        self.exp_coverage = float(format(count / len(self.All_EXP), ".3f"))
        return self.exp_coverage

    def action_constraint(self, action: Action_Class, host_info: Host_info,
                          mode):
        """
        return True if action is constrainted
        """

        if mode == UTIL.Manual:
            return None

        if action.name == self.PORT_SCAN.name:
            if host_info.port:
                return self.action_repetition

        if action.name == self.SERVICE_SCAN.name:
            if not host_info.port:
                return self.PORT_required
            if host_info.services:
                return self.action_repetition

        if action.name == self.OS_SCAN.name:
            if not host_info.port:
                return self.PORT_required
            if host_info.os:
                return self.action_repetition

        if action.name == self.WEB_SCAN.name:
            if not host_info.port:
                return self.PORT_required
            if not host_info.services:
                return self.SERVICE_required
            if host_info.web_fingerprint:
                return self.action_repetition

        if action.type == self.All_EXP[0].type:
            if not host_info.port:
                return self.PORT_required
            if action.id in self.history_set:
                return self.action_repetition

               
        return None


    @classmethod
    def test_action(cls, action_mask):
        if type(action_mask) == int:
            if action_mask < 0 or action_mask > (cls.action_space):
                logging.error("legal actions include " +
                              ",".join([a.name for a in Action.legal_actions]))
                return False
            else:
                return True
        elif type(action_mask) == str:
            if action_mask in [a.name for a in Action.legal_actions]:
                return True
            else:
                logging.error("legal actions include " +
                              ",".join([a.name for a in Action.legal_actions]))
                return False
        else:
            return False

    @classmethod
    def get_action(cls, action_mask: int):
        action_name = cls.legal_actions[action_mask].name
        return action_name

