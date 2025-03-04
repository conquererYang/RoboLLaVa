## 🔔 News
- $\color{red}{\text{[2024-09-1]}}$ We now realse the **LLaMa3+ViLD** agent codes. 
You can also download the LLaMa3+ViLD agent code here: https://pan.quark.cn/s/3d09b040086c
RoboLLaVA Model files(format safetensors): https://huggingface.co/conquererYang/robollava
We also provide robollava(finetuned with llava-llama3 & COCO dataset) agent Model files(format safetensors & gguf) here: 1.Safetensors: https://pan.quark.cn/s/89cb43851431 2.gguf: https://pan.quark.cn/s/0132e5c1dd28

## Paper Title and abstract
Active Perception Strategies for Enhanced Multimodal Integration in Robotic Grasping

Abstract— We propose a multimodal robotic agent that addresses the limitations of passive perception and fixed-model deployment through a flexible large-model architecture. This architecture enables dynamic selection of multimodal large language models (MLLMs) based on computational constraints. Building on this foundation, we introduce a logically guided active perception strategy that decides which skills (e.g., knock, weigh) to employ based on intermediate reasoning, rather than exhaustively executing all possible actions. Our work focuses on cohesive skill integration within a unified control loop, optimizing both perception and action. This allows the agent to strategically probe objects’ visual, auditory, tactile, and weight attributes for accurate material inference and robust task completion. Extensive evaluations in the Matcha [7] environment highlight the efficiency and adaptability of our method, especially in active perception and latent information inference for robotic grasping. 

Citation： 
Yang Yang, Jiankun Yang, Haibo Lu, Ruyang Liu, Huaping Liu, and Wen Gao, "Active Perception Strategies for Enhanced Multimodal Integration in Robotic Grasping," IROS, 2025.

## 🎥 Paper Introduction Video
https://github.com/user-attachments/assets/808640d9-4099-4418-a59a-e49662e8c9ed

The active Perception Strategy framework:
![figure1](https://github.com/user-attachments/assets/88808785-435f-44ad-b59e-3e54ce14f5f4)
The active perception strategy addresses the limitations of passive sensing in robotic grasping by enabling the robot to strategically collect information from complex multimodal data (Fig.3). Unlike passive perception, which relies on fixed sequences of actions, our approach allows the robotic agent to dynamically select actions based on intermediate reasoning outcomes. This flexibility is crucial for optimizing the decision-making process in environments where information is incomplete or ambiguous.
The active perception strategy is grounded in the principle of logically guided decision-making. The algorithm underlying the active perception strategy involves a sequential decision-making process.The robot operates in an environment where each object possesses both visible attributes (e.g., color, shape, location) and latent characteristics (e.g., material type). The active perception strategy employs a dynamic skill selection mechanism that prioritizes actions based on their expected utility. The robot leverages the outputs from the MLLM to assess the relevance of each potential action. For instance, if the robot identifies an object as lightweight using the ”weigh” skill, it can infer that the material is likely plastic or fiber. In this case, the robot should prioritize using the ”knock” skill for further assessment, as the tactile feedback from the ”touch” skill would likely indicate a soft or flexible texture.
Since these tactile characteristics are expected for both materials, employing the ”touch” skill would be redundant and illogical. This feedback can provide additional insights into the object’s material properties, helping to refine its understanding.  The decision to select a specific skill is guided by the outcome of the action in terms of material inference andutility function U{zi, ai}, which reflects the expected grasp execution success. Specifically, U(zi, ai) measures the degree to which the observation zi (resulting from action ai) improves the robot’s understanding of the object’s latent characteristics. For example, if the robot infers that an object is lightweight (from the ”weigh” skill), the utility of executing the ”knock” skill would be higher than the ”touch” skill, as the auditory feedback from knocking provides more discriminative information about the material.
By iteratively incorporating newly observed modalities into the MLLM, the agent refines its belief state regarding an object’s latent features, ultimately deciding when sufficient evidence has been collected to perform a grasp action
with high confidence. This formulation ensures the agent maintains a flexible approach to skill selection, enabling ondemand usage of sensors and actuators for efficient, accurate object identification and robust task completion.
## Case Study
![case1newnew](https://github.com/user-attachments/assets/70d9512c-a94e-44e0-9bc9-e3316223e69c)
![case2new](https://github.com/user-attachments/assets/762d11bd-9eae-4481-97a0-b5e43daea3dc)
## Result
![visual](https://github.com/user-attachments/assets/15512fb9-3c50-426c-a4ac-53647ee20664)
![image](https://github.com/user-attachments/assets/07262e85-fade-4818-975c-f21b3756d30f)
After introducing the active perception strategy within the flexible framework of MLLMs, the task execution process no longer exhibited skill redundancy, such as skill repetition or the use of all skills, which is a common issue with the Matcha agent. The active perception strategy enables the robotic agent to make decisions immediately after reasoning about the target material within a limited number of steps (Fig.4). Furthermore, with the active perception strategy, the robotic agent demonstrates a more focused approach to tasks (Fig.5). Agents strategically adjust skill usage to quickly complete active perception and accurately infer the correct interaction target, thereby logically accomplishing the grasping task.
## STEPS
1.Install Dependencies
environment: Ubuntu22.04 python3.9.10 cuda12.2 Nvidia535.171.04 CoppeliaSim4.4.0 qt=5.12.5
2.Coppeliasim
#NVIDIA driver和CUDA  https://blog.csdn.net/weixin_55749979/article/details/122694538
#download CoppeliaSim4.4.0-ubuntu22.04
https://blog.csdn.net/BIT_HXZ/article/details/117691451
#install CoppeliaSim4.4.0 https://blog.csdn.net/konodiodaaa/article/details/132418648
tar -xf CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz
#rename CoppeliaSim

#PATH:
```bash
sudo vim ~/.bashrc
export COPPELIASIM_ROOT=/home/yyang/CoppeliaSim
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
source ~/.bashrc

#test coppeliasim
cd CoppeliaSim
./coppeliasim
```
3.Robotic
```bash
git clone git@github.com:conquererYang/RoboLLaVa.git  #use git@github.com:conquererYang/Matcha-agent.git to get other large files(ViLD NiCOL etc.)
conda create -n matcha_new python=3.9
conda activate matcha_new
conda install qt=5.12.5
#conda list #show qt version

#install pyrep4.1.0.3 rlbench1.2.0
conda activate matcha_new
#install Pyrep-4.1.0.3 rlbench==1.2.0 https://blog.csdn.net/konodiodaaa/article/details/132418648
git clone https://github.com/stepjam/PyRep.git
cd PyRep
pip3 install -r requirements.txt
python3 setup.py install --user
pip install -e.

git clone https://github.com/stepjam/RLBench.git
cd RLBench
python3 setup.py install --user
pip install -e.


#install NICOL
cd NICOL
pip install -r requiremetns.txt #delete pyrep and rlbench in requirements(we have install them before)
cd Matcha
pip install -r requiremetns.txt

bug:#ModuleNotFoundError: No module named 'loguru'
pip install loguru
```

4.ViLD
```bash
#in another terminal
conda create -n vild python=3.9
conda activate vild
pip install -r requirements.txt

#pytorch:
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

# Download weights
gsutil cp -r gs://cloud-tpu-checkpoints/detection/projects/vild/colab/image_path_v2 ./

#open lauch_vild_server.sh (for only 1 GPU)
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 python vild_server.py ${@}  #devices=0 for desktop computer

sh lauch_vild_server.sh
```
#terminal shows:
#in browser:
* Running on all addresses (0.0.0.0) : 
0.0.0.0:8848
* Running on http://127.0.0.1:8848
* Running on  

#open browser and input the address according the terminmal output address and add the /api/vild
 
#then in ubuntu terminal return:192.168.27  "GET /api/vild HTTP/1.1" 200 -  #success

5. Sound
```bash
conda create -n sound python=3.9
conda activate sound
pip install -r requirements.txt

sudo apt-get install sox libsox-dev libsox-fmt-all  #depend library

#open train.py
sc = SoundClassifier(data_path='/home/yyang/Matcha-agent/Sound/resources', 
                     ignore=['fibre',])
python train.py #get the sound train results
#change sound path in knocking.py：SOUND_PATH = "/home/yyang/Matcha-agent/Sound/resources/"

sh lauch_sound_server.sh
```
#terminal shows:
#in browser:
* Running on all addresses (0.0.0.0) : 
0.0.0.0:8848
* Running on http://127.0.0.1:8849
* Running on http://192.168.22.27:8849 

#open browser and input the address according the terminmal output address and add the /api/sound
http://192.168.22.27:8849/api/sound
#in ubuntu return:192.168.22.27 "GET /api/sound HTTP/1.1" 200 -  #success

6.LLama3
```bash
#Install Ollama https://github.com/ollama/ollama

ollama run llama3:8b-instruct-q8_0
#or
ollama run llama3:8b
#in Browser(http://127.0.0.1:11434/) see: Ollama is running 

#for API test
npm config set registry http://mirrors.cloud.tencent.com/npm/
git clone https://github.com/ollama-webui/ollama-webui-lite.git
cd ollama-webui-lite
npm install
npm run dev
#Browser: localhost:3000 to see the webUI API

```
7.run robot agent
```bash
conda activate matcha_new
cd Matcha
pip install playsound
pip install wandb
wandb login  #imput your key
pip install pygobject
pip install requests

python main.py     #sound
python main.py -a  #sound_use_adjective

```
Citation
Yang Yang, Jiankun Yang, Haibo Lu, Ruyang Liu, Huaping Liu, and Wen Gao, "Active Perception Strategies for Enhanced Multimodal Integration in Robotic Grasping," IROS, 2025.
