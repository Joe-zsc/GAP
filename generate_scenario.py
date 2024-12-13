import os
from pprint import pprint, pformat
from datetime import datetime
import pandas as pd
import numpy as np
import random
from random import sample
import torch
import json
from pathlib import Path
import csv
from collections import Counter
'''
采样动作
'''
current_path = Path.cwd()
GatheredInfo_path = current_path / "GatheredInfo"
vul_product_path = GatheredInfo_path / "vul-product.json"
vul_description_path = GatheredInfo_path / "vul-description.json"
vul_product_uniform2_path = GatheredInfo_path / "vul-product_uniform2.json"
EXP_info_path = GatheredInfo_path / "exploit_info.json"  # 提取自msf

scenario_file_path = current_path / "scenarios"
action_file_path = current_path / "actions"
all_host = scenario_file_path / "all_scenario.json"
all_host_msf = scenario_file_path / "all_scenario_msf.json"
with open(vul_product_path, 'r', encoding='utf-8') as f:  # *********
    vul_product: dict = json.loads(f.read())
with open(vul_description_path, 'r', encoding='utf-8') as f:  # *********
    vul_description: dict = json.loads(f.read())

with open(vul_product_uniform2_path, 'r', encoding='utf-8') as f:  # *********
    vul_product_uniform: dict = json.loads(f.read())

with open(EXP_info_path, 'r', encoding='utf-8') as f:  # *********
    all_exp_info: dict = json.loads(f.read())


def save(path, data):
    with open(path, "w", encoding='utf-8') as f:
        f.write(json.dumps(data, ensure_ascii=False, indent=4))


def generate_scenario(host_number, all_target_file, scenario_file):
    with open(all_target_file, 'r', encoding='utf-8') as f:  # *********
        env_data = json.loads(f.read())
    assert host_number <= len(env_data)
    selected_hosts = random.sample(env_data, host_number)
    save(path=scenario_file, data=selected_hosts)


def generate_chain_scenario(
                            saving_scenario_file: Path,
                            all_target_file=all_host,
                            chain_length=0,
                            include_vuls=[],
                            save_sub_length=False,
                            except_vuls=[],
                            mode="ramdom"):
    assert mode in ["random", "sequence"]
    with open(all_target_file, 'r', encoding='utf-8') as f:  # *********
        env_data = json.loads(f.read())

    env_data_ = []
    if include_vuls:
        for host in env_data:
            for host_vul in host["vulnerability"]:
                if host_vul in include_vuls:
                    env_data_.append(host)
        env_data = env_data_
    if except_vuls:
        for host in env_data:
            for vul in host["vulnerability"]:
                if vul in except_vuls:
                    env_data.remove(host)
    if chain_length <= 0:
        chain_length = len(env_data)
    # scenario_file = scenario_file_path / "chain" / scenario_file
    if ".json" not in saving_scenario_file.name:
        if not os.path.exists(saving_scenario_file):
            os.mkdir(saving_scenario_file)
    else:
        if not os.path.exists(saving_scenario_file.parent):
            os.mkdir(saving_scenario_file.parent)
    assert chain_length <= len(env_data)
    if mode == "random":
        selected_hosts = random.sample(env_data, chain_length)
    elif mode == "sequence":
        selected_hosts = env_data[:chain_length]
    if save_sub_length:
        for i in range(chain_length):
            save(path=saving_scenario_file / f"subchain-{i+1}.json",
                 data=selected_hosts[:i + 1])
    else:
        save(path=saving_scenario_file, data=selected_hosts)
    return selected_hosts

def generate_single_scenario(all_target_file, scenario_path):
    with open(all_target_file, 'r', encoding='utf-8') as f:  # *********
        env_data = json.loads(f.read())
    for env in env_data:
        vul = env["vulnerability"][0]
        save_file = scenario_path / f"env-{vul}.json"
        save(path=save_file, data=[env])


def test_env(env_file=r"scenarios\large_env.json"):
    all_actions = []
    all_ip = []
    with open(env_file, 'r', encoding='utf-8') as f:  # *********
        env_data = json.loads(f.read())

        for host in env_data:
            # 漏洞无重复
            assert host["vulnerability"][0] not in all_actions
            all_actions.append(host["vulnerability"][0])

            # IP无重复
            assert host["ip"] not in all_ip
            all_ip.append(host["ip"])

            assert isinstance(host["ip"], str)
            assert isinstance(host["port"], list)
            assert isinstance(host["services"], list)
            assert isinstance(host["os"], str)
            assert isinstance(host["web_fingerprint"], list)
            assert isinstance(host["vulnerability"], list)
    print(f"scenario {env_file} generated finished")
    return all_ip, all_actions


def uniform_product(product: str):
    if product.lower().find("tomcat") != -1:
        return "Apache Tomcat"

    if product.find("https://github.com/rails/rails") != -1:
        return "Ruby on Rails"

    if product.lower().find("postgresql") != -1:
        return "PostgreSQL"

    if product.lower().find("webLogic") != -1:
        return "WebLogic Server"

    if product.lower().find("joomla") != -1:
        return "Joomla!"

    if product.lower().find("shiro") != -1:
        return "Apache Shiro"

    if product.lower().find("drupal") != -1:
        return "Drupal"

    if product.lower().find("saltstack") != -1:
        return "SaltStack Salt"

    if product.lower().find("openssl") != -1:
        return "OpenSSL"

    if product.lower().find("smb") != -1:
        return "SMB"

    if product.lower().find("libssh") != -1:
        return "libssh"

    if product.lower().find("coldfusion") != -1:
        return "Adobe ColdFusion"

    if product.lower().find("jboss") != -1:
        return "JBoss"

    if product.find("BIG-IP") != -1:
        return "BIG-IP"

    if product.lower().find("log4j") != -1 or product.find(
            "Apache Log4j") != -1:
        return "Apache Log4j"

    if product.lower() == "samba":
        return "Samba"

    if product.find("Spring Framework") != -1:
        return "Spring Framework"

    if product.find("phpMyAdmin") != -1:
        return "phpMyAdmin"

    if product.find("Spring Boot") != -1:
        return "Spring Boot"

    if product in ["Apache Struts", "struts", "Struts"]:
        return "Apache Struts"

    # if product in ["jenkins","Jenkins"]:
    #     return "Jenkins"
    if product.lower().find("jenkins") != -1:
        return "Jenkins and its plugins"
    if product in ["Confluence Server", "Confluence"]:
        return "Confluence Server"

    if product in ["Apache Solr", "Solr"] or product.find("Apache Solr") != -1:
        return "Apache Solr"

    return product


def generate_msf_exploit():
    all_exp_actions = []
    all_vul = []
    i = 0
    for exp in all_exp_info:

        ##exp字段信息
        exp_info = {}
        exp_info["description"] = exp["description"]
        exp_info["name"] = exp["fullname"]
        exp_info["description"] = exp["description"]
        exp_info["rank"] = exp["rank"]
        exp_info["available_targets"] = exp["available_targets"]
        exp_info["privileged"] = exp["privileged"]
        required_options = []
        for o in exp["options"]:
            required_option = {}
            if o["required"] == "true":
                required_option["name"] = o["name"]
                required_option["description"] = o["description"]
                required_option["display_value"] = o["display_value"]
                required_options.append(required_option)
        exp_info["required_options"] = required_options

        action = {}
        action["id"] = i
        action["type"] = "Exploit"
        action["name"] = exp["fullname"]
        action["vulnerability"] = exp["related_vulnerability"]
        # action["vul_description"]=vul_description[exp["related_vulnerability"]]
        action["exp_info"] = exp_info

        all_vul += action["vulnerability"]
        all_exp_actions.append(action)
        i += 1
    msf_exploit_actions = {}
    msf_exploit_actions["actions"] = all_exp_actions
    current_path = Path.cwd()
    action_file_path = current_path / "actions"
    save_path = action_file_path / "MSF_Exploits"
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    save(path=save_path / "actions.json", data=msf_exploit_actions)


def generate_action_set_2(Action_NUM, env_file):
    '''
    采样动作方式：basic action + 集中采样basic同类型动作 +  随机采样
    '''
    env_hostip, env_actions = test_env(env_file=env_file)
    '''
    设定动作集合
    '''
    skip_actions = ["CVE-2014-0983"]  # 黑名单，无法生成句向量
    # 读取基本动作集合
    basic_actions = []
    basic_actions_file = Path("actions/basic_actions.txt")
    for line in open(basic_actions_file, "r"):  # 设置文件对象并读取每一行文件
        vul = line.replace('\n', '')  # 将每一行文件加入到list中
        # time.sleep(0.5)
        basic_actions.append(vul)

    assert Action_NUM >= len(basic_actions) + 4

    assert len(basic_actions) == len(list(set(basic_actions)))

    # 动作集合必须要包含环境中的漏洞
    for vul in env_actions:
        assert vul in basic_actions

    # 将vulproduct中的类别统一化
    vul_product_uniform = {}
    for key, value in vul_product.items():
        vul_product_uniform[key] = uniform_product(value)

    all_vul = list(vul_product_uniform.keys())
    # 找出basic 类别
    basic_product = []
    for vul in basic_actions:
        label = vul_product_uniform[vul]
        if label not in basic_product:
            basic_product.append(label)

    # 找出
    basic_same_product_vuls = []  # 与basic同类别的漏洞
    except_basic_same_product_vuls = []  # 与basic不同类别的漏洞
    basic_same_product_vuls_labels = []
    for vul in all_vul:
        label = vul_product_uniform[vul]
        if label in basic_product:
            basic_same_product_vuls_labels.append(label)
            if vul not in basic_actions:

                basic_same_product_vuls.append(vul)

            else:
                continue
        else:
            except_basic_same_product_vuls.append(vul)
    Counter_basic_same_product_vuls_labels = Counter(
        basic_same_product_vuls_labels)

    sample_num = Action_NUM - 4 - len(basic_actions)
    if sample_num <= len(basic_same_product_vuls):
        extend_vuls = random.sample(basic_same_product_vuls, sample_num)
        all_actions = basic_actions + extend_vuls
    else:
        sample_num -= len(basic_same_product_vuls)
        extend_vuls = random.sample(except_basic_same_product_vuls, sample_num)
        all_actions = basic_actions + basic_same_product_vuls + extend_vuls

    # 无重复
    assert len(all_actions) == len(list(set(all_actions)))
    return all_actions


def generate_action_set_3(Action_NUM, env_file):
    '''
    采样动作方式：basic action + vulhub action集中采样 + 随机采样
    '''
    env_hostip, env_actions = test_env(env_file=env_file)
    '''
    设定动作集合
    '''
    skip_actions = ["CVE-2014-0983"]  # 黑名单，无法生成句向量
    # 读取基本动作集合
    basic_actions = []
    basic_actions_file = Path("actions/basic_actions.txt")
    for line in open(basic_actions_file, "r"):  # 设置文件对象并读取每一行文件
        vul = line.replace('\n', '')  # 将每一行文件加入到list中
        # time.sleep(0.5)
        basic_actions.append(vul)

    assert Action_NUM >= len(basic_actions) + 4

    candicate_vuls_file = action_file_path / "candicate_vuls.txt"
    candicate_vuls = []
    for line in open(candicate_vuls_file, "r"):  # 设置文件对象并读取每一行文件
        vul = line.replace('\n', '')  # 将每一行文件加入到list中
        if vul not in basic_actions:
            candicate_vuls.append(vul)
    # 动作集合必须要包含环境中的漏洞
    for vul in env_actions:
        assert vul in basic_actions
    candicate_vuls = list(set(candicate_vuls))
    basic_actions = list(set(basic_actions))
    first_expand_vuls = list(set(basic_actions + candicate_vuls))
    first_expand_vuls_labels = []
    for vul in first_expand_vuls:
        first_expand_vuls_labels.append(vul_product_uniform[vul])
    first_expand_vuls_labels_set = list(set(first_expand_vuls_labels))
    for vul in first_expand_vuls:
        assert vul in vul_description.keys()
    # 如果动作数量小于basic+vulhub
    if Action_NUM <= len(first_expand_vuls) + 4:
        extend_vuls = random.sample(candicate_vuls,
                                    Action_NUM - len(basic_actions) - 4)
        all_actions = basic_actions + extend_vuls
    else:
        # 根据设定的动作数量，随机补充动作空间
        sample_num = Action_NUM - len(first_expand_vuls) - 4
        all_vuls = list(vul_product_uniform.keys())
        assert len(all_vuls) == len(list(set(all_vuls)))
        sample_i = 0
        extend_vuls = []
        second_expand_vuls = []  # 采样具有同样label的vul
        execpt_second_expand_vuls = []  # 不同label的vul
        for vul in all_vuls:
            label_ = vul_product_uniform[vul]
            if label_ in first_expand_vuls_labels_set:
                if vul not in first_expand_vuls:
                    second_expand_vuls.append(vul)
                else:
                    continue
            else:
                execpt_second_expand_vuls.append(vul)
        assert len(second_expand_vuls) == len(list(set(second_expand_vuls)))
        assert len(execpt_second_expand_vuls) == len(
            list(set(execpt_second_expand_vuls)))

        if sample_num > len(second_expand_vuls):
            all_actions = first_expand_vuls + second_expand_vuls
            expand_third = random.sample(execpt_second_expand_vuls,
                                         sample_num - len(second_expand_vuls))
            all_actions += expand_third
        else:
            all_actions = first_expand_vuls + \
                random.sample(second_expand_vuls, sample_num)

    # 无重复
    assert len(all_actions) == len(list(set(all_actions)))
    return all_actions


def generate_action_set_1(Action_NUM,
                          env_file,
                          basic_actions_file=scenario_file_path /
                          "env_vuls.txt"):
    '''
    采样动作方式：basic action + 随机采样
    '''
    env_hostip, env_actions = test_env(env_file=env_file)
    '''
    设定动作集合
    '''
    skip_actions = ["CVE-2014-0983"]  # 黑名单，无法生成句向量
    # 读取基本动作集合
    basic_actions = []
    # basic_actions_file = scenario_file_path / "env_vuls.txt"
    for line in open(basic_actions_file, "r"):  # 设置文件对象并读取每一行文件
        vul = line.replace('\n', '')  # 将每一行文件加入到list中
        # time.sleep(0.5)
        basic_actions.append(vul)

    # 动作集合必须要包含环境中的漏洞
    for vul in env_actions:
        assert vul in basic_actions

    # 根据设定的动作数量，随机补充动作空间

    all_vul = []

    for vul in list(vul_product.keys()):
        if vul not in basic_actions and vul not in skip_actions:
            all_vul.append(vul)

    extend_vuls = random.sample(all_vul, Action_NUM - len(basic_actions) - 4)
    all_actions = basic_actions + extend_vuls
    return all_actions


def save_action_info(action_path, action_list, vul_product):
    if not os.path.exists(action_path):
        os.mkdir(action_path)

        with open(vul_description_path, 'r',
                  encoding='utf-8') as f:  # *********
            vul_description: dict = json.loads(f.read())
        actions_vul = {}
        actions_vul["actions"] = []
        action_label = {}
        id = 0
        vul_description_csv = action_path / "info.csv"
        # open the file in the write mode
        f = open(vul_description_csv, 'w', encoding="UTF-8")

        # create the csv writer
        writer = csv.writer(f)
        writer.writerow(["id", "CVE_ID", "Label", "Description"])
        label_selected = []
        for vul in action_list:
            vul_info = {}
            vul_info["id"] = id
            vul_info["type"] = "Exploit"
            vul_info["name"] = f"{vul}"
            vul_info["vulnerability"] = [vul]
            vul_info["exp_info"] = {}
            actions_vul["actions"].append(vul_info)

            # label = vul_product[vul]
            # label = uniform_product(label)
            # label_selected.append(label)
            # action_label[vul] = label

            # des = vul_description[vul]
            # writer.writerow([id, vul, label, des])
            id += 1
        # label_selected_counter = Counter(label_selected).most_common()
        # label_selected_counter_data = pd.DataFrame(label_selected_counter)
        # label_selected_counter_data.to_csv(action_path / "labels_count.txt",
        #                                    sep='\t',
        #                                    index=0,
        #                                    header=0)
        save(path=action_path / "actions.json", data=actions_vul)
        # save(path=action_path / "labels.json", data=action_label)
        print(f"action set {action_path} generated finished")
        f.close()


'''
设定测试环境
'''


def generate_action_set(action_num, host_num, seed=0, action_sample_mode=2):

    # generate_single_scenario(all_target_file=all_host,scenario_path=scenario_file_path/"single")
    '''
    环境和动作集检测
    '''

    # scenario_file = f"scenario-{host_num}-seed-{seed}.json"
    # scenario_file = os.path.join("scenarios", scenario_file)
    # if not os.path.exists(scenario_file):
    #     generate_scenario(all_target_file=all_host,
    #                       host_number=host_num,
    #                       scenario_file=scenario_file)
    '''
    1: 完全随机采样
    2: 优先采样靶场同类型漏洞(效果最好，默认方式)
    3: 靶场漏洞+vulhub漏洞+优先采样两者同类型
    '''
    action_name = f"Action-{action_num}-seed-{seed}-mode-{action_sample_mode}"
    action_path = action_file_path / action_name

    if action_sample_mode == 1:
        sampled_actions = generate_action_set_1(action_num, all_host)
        save_action_info(action_path=action_path,
                         action_list=sampled_actions,
                         vul_product=vul_product)
    if action_sample_mode == 2:
        sampled_actions = generate_action_set_2(action_num, all_host)
        save_action_info(action_path=action_path,
                         action_list=sampled_actions,
                         vul_product=vul_product_uniform)
    if action_sample_mode == 3:
        sampled_actions = generate_action_set_3(action_num, all_host)
        save_action_info(action_path=action_path,
                         action_list=sampled_actions,
                         vul_product=vul_product_uniform)

    # Configure.set("common", "vul_hub", action_path)


if __name__ == '__main__':
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    all_msf_vul = []
    for exp in all_exp_info:

        ##exp字段信息
        all_msf_vul += exp["related_vulnerability"]
    all_msf_vul = list(set(all_msf_vul))
    
    '''
    # ---------------------------------------------------------------------------- #
    #               run the code below if new env is added to single               #
    # ---------------------------------------------------------------------------- #
    '''
    all_envs = []
    all_envs_msf = []
    all_env_vuls = []
    env_vuls_with_exp = []
    all_single_env_path = scenario_file_path / "single"
    fake_ip = "192.168.2."
    count = 0
    for env in all_single_env_path.iterdir():
        count += 1
        with open(env, 'r', encoding='utf-8') as f:  # *********
            single_env: dict = json.loads(f.read())
        env_data = single_env[0]

        env_data["ip"] = fake_ip + str(count)
        host_vul = env_data["vulnerability"][0]
        all_env_vuls.append(host_vul)
        all_envs.append(env_data)
        if host_vul in all_msf_vul:
            env_vuls_with_exp.append(host_vul)
            all_envs_msf.append(env_data)
            ###------
            save(path=scenario_file_path / "msfexp_vul" /
                 f"env-{host_vul}.json",
                 data=[env_data])
    print(f"all host number: {count}")
    save(path=all_host, data=all_envs)
    save(path=all_host_msf, data=all_envs_msf)
    env_vuls_with_exp.sort()
    with open(scenario_file_path / "env_vuls.txt", 'w',
              encoding='utf-8') as f:  # *********
        # f.writelines(all_vuls)
        f.write('\n'.join(all_env_vuls))
    with open(scenario_file_path / "env_vuls_with_exp.txt",
              'w',
              encoding='utf-8') as f:  # *********
        # f.writelines(all_vuls)
        f.write('\n'.join(env_vuls_with_exp))

    # generate_chain_scenario(all_target_file=all_host,
    #                         scenario_file=f"msftargets-seed-{seed}",
    #                         include_vuls=all_msf_vul
    #                         )
    # generate_action_set(action_num=1000,
    #                     host_num=40,
    #                     seed=0,
    #                     action_sample_mode=1)

    # generate_msf_exploit()

    # chain_size = 40
    # scenario_file = scenario_file_path / "chain" / f"chain-{chain_size}-seed-{seed}"
    # generate_chain_scenario(all_target_file=all_host,
    #                         chain_length=chain_size,
    #                         scenario_file=scenario_file)
    