### CS419 Final Project: Group ML Bois :)
#### Team members: Abir Mehta, Gagan Jain, Parth Shastry, Tanul Gupta

Our project is to create an intelligent agent that learns to play the atari game Breakout by using a Double Deep Q-network along with experience replay.  

This readme file is a step by step guide to running the codes of our project and reproducing the results.  

We have worked on Python 3.7.5 and we highly recommend you to do the same.  
Firstly, we need to create a virtual environment. This should have all the packages along with the appropriate versions as mentioned in requirements.txt  

In order to create a new environment, one needs to have the venv package for python 3.7. For this, run -   
`sudo apt-get install python3.7-venv
`  

To create the virtual envrionment for the first time, run -   
`python3.7 -m venv RLEnv
`  

To enter the virtual environment, run -   
`source RLEnv/bin/activate
`  

Then, you need to install the packages mentioned in the virtual environment when running for the first time. This is to be done only once.  

`
pip install -upgrade pip  
python3.7 -m pip install matplotlib  
python3.7 -m pip install tensorflow==2.4.1  
python3.7 -m pip install opencv-python  
python3.7 -m pip install gym  
python3.7 -m pip install gym[atari]  
python3.7 -m pip install scikit-learn  
`  

In order to run the training, run -   
`python3.7 train.py
`  

In case you wish to resume the training from an earlier saved checkpoint, set the LOAD_FROM parameter in the file params.py to the location of the checkpoint (an example has been commented out on the same line). Note that if you do not do this, the training will start from scratch.   

If you wish to restore from an earlier checkpoint, you will have to download the checkpoint data from the drive link shown below. Note that the 'breakout-saves' folder needs to be in the same directory as this readme file.  

https://drive.google.com/drive/folders/1ryT2zOCHKS5edmTq_oN7c3NK9vEYLxF6?usp=sharing  

In case you use keyboard to interrupt training, the model till the last trained episode will be automatically saved to the 'breakout-saves' folder.  

In order to see the agent perform on the game Breakout, run -     
`python3.7 evaluation.py   
`  
This file also contains the information as to which saved checkpoint should be evaluated. So, before evaluation, you will have to download the checkpoints from the drive link or atleast train the agent for sometime to get a reasonable checkpoint. The parameter RESTORE_PATH present in the evaluation file is what needs to be set to the checkpoint path.  

In order to visualise the agent's performance and the dynamically updating Value functions, you need to run -  
`python3.7 visualize.py  
`  
This file also contains the information as to which saved checkpoint should be visualized. So, before visualization, you will have to download the checkpoints from the drive link or atleast train the agent for sometime to get a reasonable checkpoint. The parameter RESTORE_PATH present in the visualize file is what needs to be set to the checkpoint path.  
