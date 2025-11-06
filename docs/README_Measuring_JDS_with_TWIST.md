# Measuring Just Dance Score(JDS) with TWIST controller
This readme walks through the flow of testing Switch4EAI using [TWIST](https://yanjieze.com/TWIST/) as a low-level controller.
It was tested on Ubuntu 2020.4 LTS with RTX 4090 GPU.

## TWIST installation
**1**. Navigate to the forked TWIST repo
```bash
cd third_party/TWIST/
```

**2**. Checkout to the `switch4eai` branch
```bash
git checkout switch4eai
```

**3**. Install packages: 
(Note 1) Below instruction assumes you are using same conda environemt as the `switch4eai`.
(Note 2) You can also create a separate conda environment following the installation process of the [forked TWIST repo](https://github.com/whitealex95/TWIST/tree/switch4eai)

Install the below dependencies:
```bash
cd rsl_rl && pip install -e . && cd ..
cd legged_gym && pip install -e . && cd ..
pip install "numpy==1.23.0" pydelatin wandb tqdm opencv-python ipdb pyfqmr flask dill gdown hydra-core imageio[ffmpeg] mujoco mujoco-python-viewer isaacgym-stubs pytorch-kinematics rich termcolor 
pip install redis[hiredis]
pip install pyttsx3 # for voice control
cd pose && pip install -e . && cd ..
```

Manually revert the numpy and scipy version for Switch4EAI compatibility
```bash
pip install scipy=1.15.3 numpy==1.26.4
```


remember to start redis server on your computer:
```bash
redis-server --daemonize yes
```

if you wanna do sim2real, you also need to install unitree_sdk2py.
```bash
git clone https://github.com/unitreerobotics/unitree_sdk2_python.git
cd unitree_sdk2_python
pip3 install -e .
```

## TWIST Usage
**1**. Sim2sim verification:

[If this is your first time to run this script] you need to warm up the redis server by running the high-level motion server.

```bash
cd third_party/TWIST/deploy_real
python server_high_level_motion_lib.py --motion_file PATH/TO/YOUR/MOTION/FILE
```
You can just select one motion file from our motion dataset.

Then, you can run the low-level simulation server.
```bash
python server_low_level_g1_sim.py --policy_path PATH/TO/YOUR/JIT/MODEL
```
- This will start a simulation that runs the low-level controller only.
- This is because we separate the high-level control (i.e., teleop) from the low-level control (i.e., RL policy).
- You should now be able to see the robot stand still.

And now you can control the robot via high-level motion server.
```bash
python server_high_level_motion_lib.py --motion_file PATH/TO/YOUR/MOTION/FILE --vis
```

 
**2**. Sim2real verification. If you are not familiar with the deployment on physical robot, you can refer to [unitree_g1.md](./unitree_g1.md) or [unitree_g1.zh.md](./unitree_g1.zh.md) for more details.

More specifically, the pipeline for sim2real deploy is:
1. start the robot and connect the robot and your laptop via an Ethernet cable.
2. config the corresponding net interface on your laptop, by setting the IP address as `192.168.123.222` and the netmask as `255.255.255.0`.
3. now you should be able to ping the robot via `ping 192.168.123.164`.
4. then use Unitree G1's remote control to enter dev mode, i.e., press the `L2+R2` key combination.
5. now you should be able to see the robot joints in the damping state.
6. then you can run the low-level controller by:
```bash
cd third_party/TWIST/deploy_real
python server_low_level_g1_real.py --policy_path PATH/TO/YOUR/JIT/MODEL --net YOUR_NET_INTERFACE_TO_UNITREE_ROBOT
```


## Measuring Offline Just Dance Score(JDS):
You don't need Nintendo Switch to be connected to a computer for running this offline experiment.

**1**. Run low-level controller:

Sim:
```bash
cd third_party/TWIST/deploy_real
python server_low_level_g1_sim.py --policy_path PATH/TO/YOUR/JIT/MODEL
```

Real:
```bash
cd third_party/TWIST/deploy_real
python server_low_level_g1_real.py --policy_path PATH/TO/YOUR/JIT/MODEL --net YOUR_NET_INTERFACE_TO_UNITREE_ROBOT
```

**2**. Run high-level motion_lib with padded motion.
There is a time between **"A"** button press and the **start of the dancing motion**. The full information about this timing discrepancy is in [SONG_INFO.md](./SONG_INFO.md):
To handle this timing discrepancy, we have to **pad the motion file**(`*_padded.pkl`) so that the button press and the motion sequences are synchronized.
To run the motion, execute the command below and press the **"A" button at the same time** to start the song on the Switch.

```bash
python server_high_level_motion_lib.py --motion_file PATH/TO/PICKLE/FILE --vis
```

The pickled robot motion data(padded) along with the video(not-padded) is accessible in the [google drive](https://drive.google.com/drive/u/0/folders/1rOZSg6ito-b1Fir0KNuZAIwNGr9vdhb8).

These data were generated using [GVHMR](https://github.com/zju3dv/GVHMR) and [GMR](https://github.com/YanjieZe/GMR), based on mirrored video recordings of the Nintendo Switch screen so the robot can directly follow the performer’s motion.


| **Demo Sim(Screen Capture)** | **Demo Sim(Phone Recording)** |
|:---:|:---:|
| <video width="350" controls> <source src="./videos/Heart_Of_Glass_Twist_sim_offline_success.mp4" type="video/mp4"> Your browser does not support the video tag. </video> <br> *Screen Capture of running TWIST in sim using padded motion* | <video width="350" controls> <source src="./videos/Heart_Of_Glass_TWIST_sim_phone.mp4" type="video/mp4"> Your browser does not support the video tag. </video> <br> *Pressing "A" in the Switch controller is synced with starting of the padded motion. Note that the required padding may differ based on switch's internet setting* |

[Notion link](https://www.notion.so/jkim3662/Offline-TWIST_SIM-Shared-2a33ce90ec838044b7becdf3cd46ab9e)


## Measuring Online Just Dance Score(JDS):
You need Nintendo Switch to be connected to a computer for running this online experiment.

**1**. Run low-level controller:

Sim:
```bash
cd third_party/TWIST/deploy_real
python server_low_level_g1_sim.py --policy_path PATH/TO/YOUR/JIT/MODEL
```

Real:
```bash
cd third_party/TWIST/deploy_real
python server_low_level_g1_real.py --policy_path PATH/TO/YOUR/JIT/MODEL --net YOUR_NET_INTERFACE_TO_UNITREE_ROBOT
```

**2**. Run high-level motion server

This script reads socket streams from the Switch4EAI pipeline and update the imitation_observation in the redis server used by the low-level controller
```bash
cd third_party/TWIST/deploy_real
python server_switch4eai.py
```

**3**. Run the Switch4EAI's streaming module written in [README.md](../README.md#real-time-with-nintendo-switch-capture-card)

With Nintendo Switch connected to the computer:
```bash
python scripts/run_stream_to_robot.py --camera 1 --num-interp=2
```

When you run the script, you will be prompted:
```
You should Enter wait time(s) and song duration(s) for the current song:
Enter Wait Time: [enter value in seconds]
Enter Song Duration: [enter value in seconds]
Press enter again, just as you press the "a" button on the Switch Controller...
```

Wait time and song duration information are liste in [SONG_INFO](./SONG_INFO.md).
For example, when playing `Old Town Road`, you should input 4.1 for wait time and 161 for song duration.
