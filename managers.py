from pathlib import Path
from loguru import logger
import os
import uuid
import random
import openai
import numpy as np
import imageio
import json
import requests

import logging

# 配置日志记录
logging.basicConfig(level=logging.INFO) #将日志级别调整为不显示 DEBUG 信息，只显示 INFO 和更高级别的信息
#logging.basicConfig(level=logging.WARNING) #只显示 WARNING 和更高级别的信息
#logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

SKILLS = {"ask_human": 0, "knock_on": 1, "touch": 2, "pick_up": 3, "weigh": 4}
# z and quatenion
DEFAULT_POSE = np.array(
    [0.83443451, 0.76801813, -0.61692601, -0.13173638, 0.11043578, 4]
)  # the last "4" is for release


class Client:
    def __init__(self) -> None:
        self.headers = {"content-type": "application/json"}
        self.address = None

    def call(self, **kwargs):
        data = json.dumps(kwargs)
        #logger.debug(f"Sending request to {self.address} with data: {data}")
        response = requests.post(self.address, data=data, headers=self.headers)
        #logger.debug(f"Received response: {response.text}")
        result = json.loads(response.text)
        #logger.debug(f"Parsed result: {result}")
        return result


class ViLDClient(Client):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.address = "http://127.0.0.1:8848/api/vild"
        self.CATEGORY_NAMES = [
            "red block",
            "green block",
            "blue block",
            "orange block",
            "yellow block",
            "purple block",
        ]

    def call(self, category_names=None, **kwargs):
        if category_names is None:
            category_names = self.CATEGORY_NAMES
        return super().call(category_names=category_names, **kwargs)

class SoundClient(Client):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.address = "http://127.0.0.1:8849/api/sound"


class Assistant:
    def __init__(
        self, switch_vision=True, swith_sound=True, sound_use_adjective=True
    ) -> None:
        if swith_sound:
            self.sounder = SoundClient()
            self.sound_use_adjective = sound_use_adjective
        if switch_vision:
            self.vilder = ViLDClient()

    def vision(self, image_path=None, plot_on=False):
        found_objects = self.vilder.call(image_path=image_path, plot_on=plot_on)
        objects = ", ".join([k for k in found_objects])
        caption = f"[{objects}]"
        self.caption = caption
        self.found_objects = found_objects
        logger.debug(self.found_objects)
        return caption

    def sound(self, sounds):
        if len(sounds) == 0:
            sound_like = "The robot is currently not able to knock on the targeted object."
            return sound_like

        #logging.debug(f"Sounds received: {sounds}")
        sound_path = sounds[0] if sounds else "None"
        #logging.debug(f"Sound path sent: {sound_path}")

        top_probs, top_materials, top_adjectives = self.sounder.call(sound_path=sound_path)
        prob0, material0, adj0 = int(top_probs[0] * 100), top_materials[0], top_adjectives[0]
        prob1, material1, adj1 = int(top_probs[1] * 100), top_materials[1], top_adjectives[1]

        if self.sound_use_adjective:
            if prob0 > 50:
                sound_like = f"It sounds {adj0}."
            else:
                sound_like = f"It sounds {adj0} mostly and also a little bit {adj1}."
        else:
            if prob0 > 85:
                sound_like = f"It is made of {material0}."
            elif prob0 > 50:
                sound_like = f"It is probably made of {material0}."
            else:
                sound_like = f"The material cannot be certainly confirmed according to the impact sound. It could be {material0} with a {prob0}% chance, or {material1} with a {prob1}% chance."
        
        logging.info(f"[Sound]: {sound_like}")
        return sound_like

    def touch(self, touchs):
        if len(touchs) > 0:
            feeling = f"It feels {touchs[0]}."
        else:
            feeling = f"Cannot touch it."
        logger.info(f"[Feeling]: {feeling}")
        return feeling

    def weigh(self, weights):
        if len(weights) > 0:
            weighing = f"It weighs {weights[0]}."
        else:
            weighing = f"Not able to weigh it now."
        logger.info(f"[Weight]: {weighing}")
        return weighing

    def feedback(self):
        return

    def target_to_normalized_coordinates(self, target):
        invalid_target = False
        if target not in self.found_objects:
            target = random.choice([k for k in self.found_objects])
            invalid_target = True
        return (
            self.found_objects[target]["normalized_coordinates"],
            target,
            invalid_target,
        )

class Agent:
    def __init__(self, assistant: Assistant, action_tolerance=5) -> None:
        self.actions = {
            "knock_on": self.knock_on,
            "touch": self.touch,
            "weigh": self.weigh,
            "ask_human": self.ask_human,
            "pick_up": self.pick_up,
            "terminate": self.terminate,
        }
        self.assistant = assistant
        self.action_indicator = " robot." # 动作指示符
        self.action_tolerance = action_tolerance # 动作容错率

        self._clear() # 初始化变量

        
    def reset(self):
        self._clear()


    def execute(self, environment, command):  # 确认命令格式。如果不包含动作指示符，则解释命令。
        #command in a format: "AI: > robot.knock(blue block)"  

        if self.action_indicator not in command:
            return "", self.explain(command), "explain", 0, False
        
        normal = False #当目标对象无效时,随机选择其他对象避免错误，顺畅交互。true则仍执行原动作-会崩溃
        command = command.replace("[", "(").replace("]", ")")   # 分割命令获取技能和目标
        infos = command.split(self.action_indicator)[1].split("(")
        target, *_explanation = infos[1].split(")")
        if isinstance(_explanation, str):
            explanation = _explanation
        else:
            explanation = "".join(_explanation)
        
        skill = infos[0].lower()
        print("Parsed skill:", skill, "target:", target)  # Debugging: Print parsed skill and target

        if skill not in SKILLS:
            if skill in ["knockon", "knock on", "knock up", "knock_up", r"knock\_on"]:
                skill = "knock_on"
            elif skill in ["pickup", "pick up", "pick on", "pick_up", r"pick\_up"]:
                skill = "pick_up"
            elif skill in ["touchon", "touch_on", "touch up", "touch_up", r"touch\_on"]:
                skill = "touch"
            elif skill in ["weighon", "weigh_up", "weigh up", "weigh_on", r"weigh\_on"]:
                skill = "weigh"
            else:
                skill = "knock_on"
         # 将目标转换为动作
        action, chosen_target, invalid_target = self._target_to_action(target, skill)
        if self.many_duplicates:
            skill = "pick_up"
            reason = "Too many duplicated actions."
        elif self.invalid_count >= self.action_tolerance:
            skill = "pick_up"
            reason = "Too many invalid actions."
        else:
            if invalid_target:
                reason = "Invalid target."
                self.invalid_count += 1
            else:
                normal = True
        # 执行相应技能
        skill_func = self.actions[skill]
        # print(f"Executing skill function {skill_func} with action {action}...")
        # if skill_func is None:
        #     print("Error: Skill function not found for skill:", skill)
        #     return "Error", None, "error", 0, True
        logger.debug(f"Carrying out skill {skill_func} on action {action} ...")
        description, explanation, reward, done = skill_func(environment, action)
        if normal:
            return description, explanation, skill, reward, done
        # 处理异常情况的说明
        skill_natural = skill.replace("_", " ")
        description_pre = (
            f"Human: {reason} Randomly {skill_natural} the {chosen_target} instead."
        )
        description = description.replace("Human:", "")
        description = description_pre + description
        # 计算重复动作
        if description not in self.executions:
            self.executions[description] = 0
        else:
            self.executions[description] += 1
            if self.executions[description] >= self.action_tolerance:
                self.many_duplicates = True
            else:
                pass
        return description, explanation, skill, reward, done 


    def knock_on(self, environment, action):
        action = self._update_action_skill(action, skill="knock_on")
        env = environment.env
        obs, reward, terminate, info = env.step(action)
        sounds = info["sounds"]
        sound_like = self.assistant.sound(sounds)
        description = f"Human: {sound_like}\nAI:"
        return description, None, reward, False or terminate
    
    def pick_up(self, environment, action):
        env = environment.env
        action = self._update_action_skill(action, skill="pick_up")
        obs, reward, terminate, info = env.step(action)
        description = "Human: Explain why.\nAI:"
        return description, None, reward, True or terminate

    def explain(self, command):
        logger.info(command)
        description = "Human: go on.\nAI:"
        return description, command, 0, False

    def vision(self, *args, **kwargs):
        return self.assistant.vision(*args, **kwargs)

    def touch(self, environment, action):
        action = self._update_action_skill(action, skill="touch")
        env = environment.env
        obs, reward, terminate, info = env.step(action)
        touchs = info["touchs"]
        feeling = self.assistant.touch(touchs)
        description = f"Human: {feeling}\nAI:"
        return description, None, reward, False or terminate

    def ask_human(self, target):
        answer = self.assistant.feedback()
        return answer, None, 0, False

    def weigh(self, environment, action):
        action = self._update_action_skill(action, skill="weigh")
        env = environment.env
        obs, reward, terminate, info = env.step(action)
        weights = info["weights"]
        feeling = self.assistant.weigh(weights)
        description = f"Human: {feeling}\nAI:"
        return description, None, reward, False or terminate

    def terminate(self):
        reward = 0
        return "", None, reward, True

    def _clear(self):
        self.invalid_count = 0
        self.executions = {}
        self.many_duplicates = False

    def _target_to_action(self, target, skill):
        (
            normmalized_coordinates,
            target,
            invalid_target,
        ) = self.assistant.target_to_normalized_coordinates(target)
        return (
            np.r_[normmalized_coordinates, DEFAULT_POSE[:-1], SKILLS[skill]],  # 组合进标准化的坐标和预定义的动作姿势
            target,
            invalid_target,
        )

    def _update_action_skill(self, action, skill):  # 更新动作的技能部分
        action[-1] = SKILLS[skill]
        return action


class ChatEnvironment:
    def __init__(
        self,
        env_cls,
        mode="test",
        headless=False,
        temp_directory="./temp",
        debug=True,
        render_mode=None,
    ) -> None:
        if render_mode == "None":
            render_mode = None
        env = env_cls(
            observation_mode="vision", headless=headless, render_mode=render_mode
        )
        env.set_mode(mode)  # either 'train' or 'test'
        env.set_random(not debug)
        self.env = env
        self.temp_directory = Path(temp_directory).absolute()
        self.temp_files = []
        self.instruction = None

    def reset(self):
        try:
            obs = self.env.reset()
        except RuntimeError:
            logger.warning("Bad initialization, reset.")
        except Exception as e:
            raise e
        # image = obs["front_rgb"]
        self.instruction = self.env.info["instruction"]
        return

    def instruct(self):
        return self.instruction

    def instruct_with_caption(self, caption=None):
        if caption is None:
            instruction = f'Human: "{self.instruction}".\n'
        else:
            instruction = (
                f'Human: "{self.instruction}" in the scene that contains {caption}.\n'
            )
        return instruction

    def render(self):
        env = self.env
        image = env.states["front_rgb"]
        temp_image_name = str(uuid.uuid4())[-8:] + ".jpg"
        if not self.temp_directory.exists():
            self.temp_directory.mkdir()
        temp_image_path = str(self.temp_directory / temp_image_name)
        imageio.imwrite(temp_image_path, image)
        self.temp_files.append(temp_image_path)
        return temp_image_path

    def clean_up(self):
        for temp_file in self.temp_files:
            os.remove(temp_file)
            logger.warning(f"Removed {temp_file}.")
        self.temp_files = []
        return


LLM_CACHE = {}


class LLM:
    def __init__(
        self,
        engine="llama3:8b",
        openai_api_base="http://127.0.0.1:11434/api/generate",
        openai_api_key="EMPTY",
        prompt_path="/home/yyang/Matcha-agent/prompts.txt",
        max_tokens=1024,
        temperature=0.3,
        top_p=0.85,
        frequency_penalty=0.1,
        presence_penalty=0.3
    ) -> None:
        self.api_url = openai_api_base.rstrip('/')
        self.api_key = openai_api_key  # 并未使用
        if prompt_path is not None:
            with open(prompt_path) as f:
                prompt = f.read() + "\n"
        else:
            prompt = ""
        self._prompt = prompt
        self.engine = engine
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty

    def reset(self):
        pass

    def feed(self, prompt=""):
        if "explain" in prompt.lower():
            max_tokens = 128
        else:
            max_tokens = 16
        prompt = self._prompt + prompt
        response = self.call_model(prompt, max_tokens=max_tokens)
        command = response.split("\n")[0]
        return command + "\n"

    def call_model(self, prompt, max_tokens=1024):  # 调用方法定义在 LLM 类内部
        headers = {
            'Content-Type': 'application/json',
        }
        data = {
            "model": self.engine,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,        # nucleus sampling
            "frequency_penalty": self.frequency_penalty,    # Penalty for repeated terms
            "presence_penalty": self.presence_penalty   # Encouragement for introducing new terms
        }
      #  print(f"Making POST request to {self.api_url} with headers={headers} and data={data}") #生成的对话

        try:
            response = requests.post(self.api_url, headers=headers, data=json.dumps(data), stream=True)
            result = ""
           # print(f"Received response with status code {response.status_code}") #是否成功接入api 200/404
            if response.status_code == 200:
                #print(f"Response content: {response.text}") #打印llm谈话信息
                for line in response.iter_lines():
                    if line:
                        decoded_line = line.decode('utf-8')
                        try:
                            json_line = json.loads(decoded_line)
                            result += json_line.get('response', "")
                        except json.JSONDecodeError as e:
                            print(f"JSON Decode Error: {e}, line content: {decoded_line}")
                return result
            else:
                print(f"Error response content: {response.text}")
                raise Exception(f"Error {response.status_code}: {response.text}")
        except requests.exceptions.RequestException as e:
            print(f"HTTP request failed: {e}")
            raise Exception(f"HTTP request failed: {e}")

    def gpt3_call(self, prompt, max_tokens=128, logprobs=1, echo=False):
        return self.call_model(prompt, max_tokens=max_tokens)


class FakeLLM(LLM):
    def __init__(self, *args, **kwargs):
        self.count = 0
        self.diversity = [
            "> robot.knock_on(random object)\n",
            "> robot.touch(random object)\n",
            "> robot.weigh(random object)\n",
            "> robot.weigh(random object)\n",
            "> robot.touch(random object)\n",
            "> robot.touch(random object)\n",
            "> robot.touch(random object)\n",
            "> robot.knock_on(random object)\n",
            "> robot.weigh(random object)\n",
            "> robot.knock_on(random object)\n",
        ]

    def reset(self):
        self.count = 0

    def feed(self, *args, **kwargs):
        rtn = self.diversity[self.count]
        self.count += 1
        return rtn
