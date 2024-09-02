## 🔔 News
- $\color{red}{\text{[2024-09-1]}}$ We now realse the **LLaMa3+ViLD** agent codes. the full codes coming soon.

RoboLLaVA Model files(format safetensors & gguf): https://huggingface.co/conquererYang/robollava
RoboLLaVA in ollama: https://ollama.com/yyang/robollava


## 🎥 Demo Video
###Distinct
https://www.youtube.com/playlist?list=PLIJnmuEVkn7LE1vMhbhY2SCgFni8Dtkwx
Includes: 1.Vicuna+ViLD 2.LLaMA3+ViLD 3.LLaVA-llama3 4.RoboLLaVA
![image](https://github.com/user-attachments/assets/64144eb5-fe38-479d-8fdc-b65560bf7be4)
###Indistinct
https://www.youtube.com/playlist?list=PLIJnmuEVkn7LFeiShF2Q-u9KLvlt_POZY
Includes: 1.Vicuna+ViLD 2.LLaMA3+ViLD 3.LLaVA-llama3 4.RoboLLaVA
![image](https://github.com/user-attachments/assets/152b24e1-fd78-4922-b747-789f2ac6e326)


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
git clone git@github.com:conquererYang/RoboLLaVa.git  
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
"RoboLLaVA:Enhancing Interactive Multimodal Perception and Robot Decision-Making with LLaVa"
