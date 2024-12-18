
from defination import Host_info, Env_data, Action_Result


class OSScan:
    

    def __init__(self, target_info: Host_info, env_data: Env_data = None):
        self.target_ip = target_info.ip
        self.target_info = target_info
        self.port = self.target_info.port
        self.os = ""
        self.env_data = env_data
        self.simulated = True # initiaized to True
    def act(self, mode=0):
        
        self.simulated = True
        os = self.simulate_act()
        
        self.os = os
        self.target_info.os = os

        if self.os:
            result = Action_Result(
                success=True,
                type="OS Scan Success",
                message=self.os,
            )
        else:
            result = Action_Result(
                success=False,
                type="OS Scan Failed",
            )

        return result

    def simulate_act(self):

        if self.env_data.ip == self.target_ip:
            return self.env_data.os
        return

    