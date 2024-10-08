from paramiko import SSHClient, SFTPClient, AutoAddPolicy, Transport
import time
import json
import re
from rich import print
from pathlib import Path


class SSH:
    def __init__(self, ip_address: str, username: str, password: str, port: int = 22):
        """
        :param ip_address:远程ip地址
        :param username:用户名
        :param password:密码
        :param port:端口号,默认22
        """
        self.ip = ip_address
        self.username = username
        self.password = password
        self.port = port
        self.__client = SSHClient()
        self.connect()
        self.channel = self.__client.invoke_shell()

    def connect(self):

        self.__client.set_missing_host_key_policy(AutoAddPolicy())
        self.__client.connect(self.ip, self.port, self.username, self.password)

    def close(self):
        self.__client.close()

    def invoke_shell(self, command):

        self.channel.sendall(command + "\r\n")
        time.sleep(2)
        # while True:
        #     if self.channel.recv_ready():
        #         print('test')
        #         outbuf = self.channel.recv(65535)
        #         if len(outbuf) == 0:
        #             raise EOFError("Channel stream closed by remote device.")
        #         output += outbuf.decode("utf-8", "ignore")
        #         print(outbuf)

    def execute(self, command: str, wait_time=1, sudo=False):

        if not sudo:
            std_in, stdout, stderr = self.__client.exec_command(command=command)
        else:
            std_in, stdout, stderr = self.__client.exec_command(
                command=f"echo {self.password}| sudo -S {command}"
            )
        time.sleep(wait_time)
        result = stdout.read().decode("utf-8")
        result2 = result.rstrip("\n")
        # print(result)
        return result2

    def upload_file(self, local_file_path: str, remote_file_path: str):

        sftp: SFTPClient = self.__client.open_sftp()
        try:
            sftp.put(localpath=local_file_path, remotepath=remote_file_path)
            print(f"file:{local_file_path} upload success！")
            return True
        except Exception as e:
            print(
                f"upload file file,please check whether the file path is correct!\nerror massage：{e} "
            )
            return False

    def download_file(self, remote_file_path: str, local_save_path):

        sftp: SFTPClient = self.__client.open_sftp()
        try:
            sftp.get(remotepath=remote_file_path, localpath=local_save_path)
            print(f"file:{remote_file_path} download success!")
        except Exception as e:
            print(
                f"upload file file,please check whether the file path is correct!\nerror massage：{e} "
            )

    def get_shell(self):

        while True:
            command = input(f"{self.ip}@{self.username}$:")
            if command.__eq__("quit"):
                break
            self.execute(command=command)


class Vulhub_operator:

    def __init__(self, target_ip, vulhub_path, username, password, ssh_port="22"):
        self.SSH = SSH(
            ip_address=target_ip,
            port=ssh_port,
            username=username,
            password=password,
        )
        self.vulhub_path_file = Path("GatheredInfo/vul_to_vulhubpath.json")
        self.vulhub_path = Path(vulhub_path)
        self.password = password
        
    def open_vul_docker(self, vul):
        """
        password: user password of the vulhub host
        vulhub_path: path of vulhub file
        vul: the target vulnerability
        """
        result = self.SSH.execute(f"cd {str(self.vulhub_path)}")
        if result:
            print(f"[red]{str(self.vulhub_path)} not found[/]")
            return False
        with open(self.vulhub_path_file, "r", encoding="utf-8") as f:  # *********
            msf_vulhub_path = json.loads(f.read())
        try:
            path = str(self.vulhub_path / msf_vulhub_path[vul][0])
            print(f"[green]find the Vulhub path of {vul}: {path}")
        except:
            print(f"[red]failed to find the Vulhub path of {vul}")
            return False

        """
        # ---------------------------------------------------------------------------- #
        #                              close docker                              #
        # ---------------------------------------------------------------------------- #
        """

        # result=ssh.execute(f"echo {args.password} |sudo -S docker ps -q")
        result = self.SSH.execute(f"docker ps -q", sudo=True)
        all_container_id = result.split("\n")
        for id in all_container_id:
            if id:
                result = self.SSH.execute(
                    f"echo {self.password} |sudo -S docker stop {id}"
                )
                if result == id:
                    print(f"[green]success stop the running image {id}")
        result = self.SSH.execute(f"docker ps -q", sudo=True)
        assert not result, result

        """
        # ---------------------------------------------------------------------------- #
        #                                   open target                                   #
        # ---------------------------------------------------------------------------- #
        """

        result = self.SSH.execute(
            f"docker-compose -f {path}/docker-compose.yml up -d", wait_time=3, sudo=True
        )
        # print(result)
        result = self.SSH.execute(f"docker ps -q", sudo=True)

        if result:
            # print(result)
            print(f"[green]success start the running image {result} of {vul}")
            result = self.SSH.execute(f"docker ps", sudo=True)
            print(result)
            time.sleep(10)
            return True
        else:
            print(f"[red]failed to start the running image {result} of {vul}")
            return False


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-t",
        "--target",
        default="192.168.145.131",
        help="target ip, e.g. 127.0.0.1 or 127.0.0.1/24",
    )
    parser.add_argument(
        "--port", default="22"
    )
    parser.add_argument(
        "--username"
    )
    parser.add_argument(
        "--password"
    )
    parser.add_argument(
        "--vulhub_path", default="/home/kali/vulhub", help="path of vulhub"
    )
    parser.add_argument("--vul", default="CVE-2021-3129")

    args = parser.parse_args()

    ssh = SSH(
        ip_address=args.target,
        port=args.port,
        username=args.username,
        password=args.password,
    )

    vulhub = Vulhub_operator(
        password=args.password,
        vulhub_path=args.vulhub_path,
        target_ip=args.target,
        username=args.username,
        ssh_port=args.port,
    )

    vulhub.open_vul_docker(vul=args.vul)