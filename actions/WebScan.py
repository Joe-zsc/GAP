import sys, os
import tempfile
import json
import re
from loguru import logger as logging
from pprint import pprint, pformat

curr_path = os.path.dirname(__file__)
parent_path = os.path.dirname(curr_path)
sys.path.append(parent_path)  # add current terminal path to sys.path
sys.path.append(curr_path)  # add current terminal path to sys.path
from util import Configure, UTIL, Well_known_ports
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
        if mode in [UTIL.Train_Simulate, UTIL.Eval_Simulate] or self.env_data.web_fingerprint:
            self.simulated = True
            self.info = self.simulate_act()
        else:
            logging.success(f"----- Performing Webfingerprint Scan -----")
            self.simulated = False
            """
            # ----------------------- 1.find related port / services ----------------------- #
            """
            web_ports = []

            for i in range(len(self.target_info.services)):
                port = self.target_info.port[i]
                if self.target_info.services[i].lower().find("https") != -1:
                    web_ports.append((port, "HTTPS"))
                    continue
                if self.target_info.services[i].lower().find("http") != -1:
                    web_ports.append((port, "HTTP"))
                    continue
                if int(port) >= 1024 or port in ["80", "443"]:
                    check_result = UTIL.check_web_service(
                        port=port, host=self.target_ip
                    )
                    if check_result == "HTTP":
                        web_ports.append((port, "HTTP"))
                    elif check_result == "HTTPS":
                        web_ports.append((port, "HTTPS"))

            for web_port in web_ports:
                if web_port[1] == "HTTPS":
                    self.web_scan(port=web_port[0], is_https=True)
                elif web_port[1] == "HTTP":
                    self.web_scan(port=web_port[0])
                # else:
                # if self.curl(port=port):
                #     continue
                # else:
                #     self.curl(port=port, is_https=True)

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
            if self.simulated:
                result = Action_Result(
                    success=False,
                    type="Web Scan failed",
                )
            else:
                if not web_ports:
                    result = Action_Result(
                        success=False,
                        type="Web Scan Failed",
                        message=f"No found opened web ports",
                    )
                else:
                    result = Action_Result(
                        success=False,
                        type="Web Scan Failed",
                        message=f"Web ports {pformat(web_ports)} opened, but not fingerprint information",
                    )
        return result

    def simulate_act(self):

        if self.env_data.ip == self.target_ip:
            return self.env_data.web_fingerprint
        return []

    def whatweb(self, path, level=1):
        
        base_command = f"whatweb -a {level} --colour=never "
        command = base_command + path

        if self.target_info.pivot:
            command = "proxychains " + command
        """
        创建临时文件存放json结果
        """
        temp = tempfile.NamedTemporaryFile(suffix=".json")
        command = command + f" --log-json={temp.name}"

        status, result = UTIL.exec_shell_command(command)
        loc = result.find("200 OK")
        if loc != -1:
            scan_info = result[loc - 1 :]
            if scan_info:
                filtered_info = self.filtered_info(scan_info)
                if len(filtered_info) > self.max_info_length:
                    self.max_info_length = len(filtered_info)
                    self.info = scan_info
                    self.fliter_info = filtered_info

                    with open(temp.name, "r", encoding="utf-8") as f:  # *********
                        json_info_ = json.loads(f.read())
                        json_info = ""
                        for info in json_info_:
                            if info["http_status"] != 200:
                                continue
                            json_info = info
                        self.json_info = [json_info]
                    # print(self.info)
                    temp.close()

    def web_scan(self, port="80", is_https=False, level=1):
        logging.info(f"Start scanning webfingerprint...")
        scan_info = ""
        base_url = self.target_ip + ":" + port
        if is_https:
            url = "https://" + base_url
        else:
            url = "http://" + base_url

        self.whatweb(path=url,level=level)
        
        if not self.fliter_info:
            logging.warning("Start scanning the web server for directories...")
            possiable_path = self.dirb(url=url)
            
            for path in possiable_path:
                self.whatweb(path=path,level=level)
        if self.fliter_info and self.json_info:
            return True
        else:
            return False

    def dirb(self, url):
        web_paths=[]
        command = f"dirb {url} -S -r"
        status, result = UTIL.exec_shell_command(command)
        pattern= "FOUND: ([0-9])"
        found_num=int(re.findall(pattern, result)[0])
        
        lines=result.split('\n')
        for line in lines:
            if line.find("CODE:200")!=-1:
                web_paths += re.findall(self.url_re_pattern, line)
        return web_paths



    def filtered_info(self, info):

        filtered_info = ""
        info = info.replace("[200 OK]", "")
        info = info.split(",")
        for info_str in info:
            if info_str.find("Country") != -1:
                continue
            elif info_str.find("IP") != -1:
                continue
            elif info_str.find("Content-Language") != -1:
                continue
            else:
                filtered_info += info_str
        return filtered_info


