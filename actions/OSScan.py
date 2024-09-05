from util import Configure, UTIL
from defination import Host_info, Env_data, Action_Result
import nmap
import re
from loguru import logger as logging
import os


# try:
#     from NLP_Module.NER_Module.interact import get_ner
# except Exception as e:
#     logging.error("Error: NER module load failed: " + str(e))
class OSScan:


    def __init__(self, target_info: Host_info, env_data: Env_data = None):
        self.target_ip = target_info.ip
        self.target_info = target_info
        self.port = self.target_info.port
        self.os = ""
        self.env_data = env_data
        self.simulated = True # initiaized to True
    def act(self, mode=0):
        if mode != UTIL.Manual and self.env_data.os:
            self.simulated = True
            os = self.simulate_act()
        else:
            self.simulated = False
            logging.success(f"----- Performing OS Scan -----")
            os = self.real_act()
            self.env_data.os = os
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

    def real_act(self):
        if self.target_info.pivot:
            os = self.pivot_scan(target_ip=self.target_ip)
        else:
            os = self.os_scan(target_ip=self.target_ip)

        return os

    def clear_info(self, os_info):
        os = ""
        if os_info.find("Linux") != -1:
            os = "Linux"
        elif "indows" in os_info:

            os = re.findall(self.os_pattern, os_info, re.IGNORECASE)
            os = " or ".join(os)
            # os_ner = get_ner(os_info)
            # for item in os_ner:
            #     if item[1] == 'os':
            #         os_name = item[0]
            #     if item[1] == 'version':
            #         os_version = item[0]
            # os = os_name + ' ' + os_version
        elif "22" in self.target_info["port"]:
            os = "Linux"
        else:
            os = os_info
        return os

    def os_scan(self, target_ip, timeout=0):
        result = ""
        import subprocess

        scanner = nmap.PortScanner()
        password = UTIL.password
        command = f"nmap -oX - -O {target_ip}"
        args = f"echo {password} |sudo -S {command}"
        p = subprocess.Popen(
            args,
            shell=True,
            bufsize=100000,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if timeout == 0:
            (scanner._nmap_last_output, nmap_err) = p.communicate()
        else:
            try:
                (scanner._nmap_last_output,
                 nmap_err) = p.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                p.kill()
                raise nmap.PortScannerTimeout("Timeout from nmap process")

        nmap_err = bytes.decode(nmap_err)
        nmap_err_keep_trace = []
        nmap_warn_keep_trace = []
        if len(nmap_err) > 0:
            regex_warning = re.compile("^Warning: .*", re.IGNORECASE)
            for line in nmap_err.split(os.linesep):
                if len(line) > 0:
                    rgw = regex_warning.search(line)
                    if rgw is not None:
                        nmap_warn_keep_trace.append(line + os.linesep)
                    else:
                        nmap_err_keep_trace.append(nmap_err)

        scan_result = scanner.analyse_nmap_xml_scan(
            nmap_xml_output=scanner._nmap_last_output,
            nmap_err=nmap_err,
            nmap_err_keep_trace=nmap_err_keep_trace,
            nmap_warn_keep_trace=nmap_warn_keep_trace,
        )
        os_info = scan_result["scan"][self.target_ip]["osmatch"][0]["name"]
        # os_info = self.clear_info(os_info)  #! attention
        return os_info

    def pivot_scan(self, target_ip, timeout=0):


        import subprocess

        scanner = nmap.PortScanner()
        result = ""
        if self.port == None:
            logging.error("service scan error : please scan the ports first")
            return result
        ports_remain = ",".join(self.port)

        command = f"proxychains nmap -oX - -sV -sT {target_ip} -p {ports_remain}"

        p = subprocess.Popen(
            command,
            shell=True,
            bufsize=100000,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if timeout == 0:
            (scanner._nmap_last_output, nmap_err) = p.communicate()
        else:
            try:
                (scanner._nmap_last_output,
                 nmap_err) = p.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                p.kill()
                raise nmap.PortScannerTimeout("Timeout from nmap process")

        nmap_err = bytes.decode(nmap_err)
        nmap_err_keep_trace = []
        nmap_warn_keep_trace = []
        if len(nmap_err) > 0:
            regex_warning = re.compile("^Warning: .*", re.IGNORECASE)
            for line in nmap_err.split(os.linesep):
                if len(line) > 0:
                    rgw = regex_warning.search(line)
                    if rgw is not None:
                        nmap_warn_keep_trace.append(line + os.linesep)
                    else:
                        nmap_err_keep_trace.append(nmap_err)

        scan_result = scanner.analyse_nmap_xml_scan(
            nmap_xml_output=scanner._nmap_last_output,
            nmap_err=nmap_err,
            nmap_err_keep_trace=nmap_err_keep_trace,
            nmap_warn_keep_trace=nmap_warn_keep_trace,
        )
        # logging.info(pformat(scan_result))

        os_support_port = ["22", "139", "445"]
        result = "unknown"
        for key, value in scan_result["scan"][self.target_ip]["tcp"].items():
            port = str(key)
            cpe = value["cpe"]
            if cpe and (port in os_support_port):
                if cpe.find("windows") != -1:
                    result = "Windows"
                elif cpe.find("linux") != -1:
                    result = "Linux"
                elif "22" in self.target_info["port"]:
                    result = "Linux"

        return result
