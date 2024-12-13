import sys, os

curr_path = os.path.dirname(__file__)
parent_path = os.path.dirname(curr_path)
sys.path.append(parent_path)  # add current terminal path to sys.path
sys.path.append(curr_path)  # add current terminal path to sys.path
from defination import Host_info, Env_data, Action_Result


class WebScan:
    url_re_pattern = r"\b(?:https?|ftp)://[\w.-]+(?:\.[\w\.-]+)+[\w\-\._~:/?#[\]@!\$&'\(\)\*\+,;=.]+"
    def __init__(self, target_info: Host_info, env_data: Env_data = None):
        self.target_ip = target_info.ip
        self.target_info = target_info
        self.info = []
        self.fliter_info = []
        self.json_info = {}
        self.env_data = env_data
        self.max_info_length = 0
        self.simulated = True # initiaized to True
    def act(self, mode=0):
        
        self.simulated = True
        self.info = self.simulate_act()
        
        self.fliter_info = self.info

        self.target_info.web_fingerprint = self.fliter_info

        if not self.env_data.web_fingerprint:
            self.env_data.web_fingerprint = self.fliter_info
            self.env_data.web_fingerprint_component = self.json_info

        if self.fliter_info:
            result = Action_Result(
                success=True,
                type="Web Scan Success",
                message=self.fliter_info,
            )
        else:
            
            result = Action_Result(
                success=False,
                type="Web Scan failed",
            )
            
        return result

    def simulate_act(self):

        if self.env_data.ip == self.target_ip:
            return self.env_data.web_fingerprint
        return []

    
    


