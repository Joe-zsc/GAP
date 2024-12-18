
from defination import Host_info, Env_data,Action_Result

from loguru import logger as logging

class PortScan:



    def __init__(self, target_info: Host_info, env_data: Env_data = None):
        self.target_ip = target_info.ip
        self.target_info = target_info
        self.port_list = []
        self.env_data = env_data
        self.arguments = "-Pn"
        self.simulated = True # initiaized to True
    def act(self,mode=0):

        port_list = self.simulate_act()
        self.simulated=True
        
        self.target_info.port = port_list
        self.port_list = port_list
        
        if self.port_list:
            result=Action_Result(
            success=True,
            type="Port Scan Success",
            message=self.port_list, 
        )
            
        else:
            result=Action_Result(
            success=False,
            type="Port Scan Failed",
        )
        return result

    def simulate_act(self):

        if self.env_data.ip == self.target_ip:
            return self.env_data.port
        return []

    