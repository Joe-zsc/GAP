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
        if mode != UTIL.Manual and self.env_data.services:
            services_list = self.simulate_act()
            self.simulated = True
        else:
            logging.success(f"----- Performing Service Scan -----")
            self.simulated = False
            services_list = self.real_act()
            self.env_data.services = services_list

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

    def real_act(self):
        service_list = []
        port_list = []

        ports_remain = ",".join(self.port.copy())
        port_list = self.port.copy()
        if self.target_info.pivot:
            scan_result = self.pivot_scan(self.target_ip, ports_remain)
        else:

            scanner = nmap.PortScanner()
            scan_result = scanner.scan(hosts=self.target_ip,
                                       ports=ports_remain,
                                       arguments=self.arguments)

        for key, value in scan_result["scan"][self.target_ip]["tcp"].items():
            if value["state"] == "open":
                port = str(key)
                if not self.port:
                    port_list.append(port)
                """
                #1 精简信息 or 精确信息
                """
                if not self.exact:
                    service = value["name"]
                    if port in Well_known_ports.keys():
                        service_list.append(Well_known_ports[port])
                        continue
                    if service.find("ssl/http") != -1:
                        service = "HTTPS"
                    elif service.find("http-") != -1:
                        service = "HTTP"
                    else:
                        for key, value in Well_known_ports.items():
                            if service.lower() == value.lower():
                                service = value
                                break

                else:
                    service = (value["name"] + " " + value["product"] + " " +
                               value["version"])

                service_list.append(service)

        self.port = port_list
        return service_list

    def pivot_scan(self, target_ip, ports_remain, timeout=0):
        import subprocess

        scanner = nmap.PortScanner()

        command = (
            f"proxychains nmap -oX - {self.arguments} {target_ip} -p {ports_remain}"
        )

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

        return scanner.analyse_nmap_xml_scan(
            nmap_xml_output=scanner._nmap_last_output,
            nmap_err=nmap_err,
            nmap_err_keep_trace=nmap_err_keep_trace,
            nmap_warn_keep_trace=nmap_warn_keep_trace,
        )
