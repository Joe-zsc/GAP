import sys, os
from util import Configure, UTIL, Well_known_ports
import nmap  # pip install python-nmap
import re
from defination import Host_info, Env_data, Action_Result
from loguru import logger as logging


class ServicesScan:

    def __init__(self, target_info: Host_info, env_data: Env_data = None):
        self.target_ip = target_info.ip
        self.target_info = target_info
        self.port = self.target_info.port
        self.arguments = "-Pn -sV"
        self.services_list = []
        self.env_data = env_data
        self.exact = True
        self.simulated = True # initiaized to True
    def act(self, mode=0):

        services_list = self.simulate_act()
        self.simulated = True
        

        self.target_info.services = services_list
        self.services_list = services_list

        if self.services_list:
            result = Action_Result(
                success=True,
                type="Services Scan Success",
                message=self.services_list,
            )
        else:
            result = Action_Result(
                success=False,
                type="Services Scan Failed",
            )

        return result

    def simulate_act(self):

        if self.env_data.ip == self.target_ip:
            return self.env_data.services
        return []

    
