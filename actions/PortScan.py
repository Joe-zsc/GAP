from util import Configure, UTIL
import re
import sys
import os
import time
from defination import Host_info, Env_data,Action_Result
import nmap #pip install python-nmap
import re
from loguru import logger as logging

class PortScan:

    support_ports_str = Configure.get("Support", "port").strip().replace(" ", "")
    support_ports = support_ports_str.split(",")  # list

    def __init__(self, target_info: Host_info, env_data: Env_data = None):
        self.target_ip = target_info.ip
        self.target_info = target_info
        self.port_list = []
        self.env_data = env_data
        self.arguments = "-Pn"
        self.simulated = True # initiaized to True
    def act(self,mode=0):
        if mode != UTIL.Manual and self.env_data.port:
            port_list = self.simulate_act()
            self.simulated=True
        else:
            logging.success(f"----- Performing Port Scan -----")
            self.simulated=False
            port_list = self.real_act()
            self.env_data.port=port_list
            
        
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

    def real_act(self):

        scanner = nmap.PortScanner()
        port_list = []
        services_list = []
        if self.target_info.pivot:
            scan_result = self.pivot_scan(target_ip=self.target_ip, arguments=self.arguments)
        else:
            scan_result = scanner.scan(
                hosts=self.target_ip,
                ports=",".join(self.support_ports),
                arguments=self.arguments,
            )
        for key, value in scan_result["scan"][self.target_ip]["tcp"].items():
            if value["state"] == "open":
                port_list.append(str(key))

        return port_list

    def pivot_scan(self, target_ip,  timeout=0):

        import subprocess

        scanner = nmap.PortScanner()

        ports_remain = ",".join(PortScan.support_ports)
        command = f"proxychains nmap -oX - {self.arguments} {target_ip} -p {ports_remain}"

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
                (scanner._nmap_last_output, nmap_err) = p.communicate(timeout=timeout)
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

        return scanner.analyse_nmap_xml_scan(
            nmap_xml_output=scanner._nmap_last_output,
            nmap_err=nmap_err,
            nmap_err_keep_trace=nmap_err_keep_trace,
            nmap_warn_keep_trace=nmap_warn_keep_trace,
        )
