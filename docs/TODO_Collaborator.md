To. Collaborators

Here are the things we'd like you to do:


**1.** Record G1’s Dance (Front View)
- Please record the robot's motion using your camera(a phone is fine), and upload the video (.mp4 format preferred) in [GD](https://drive.google.com/drive/u/0/folders/1m7GYZQWQi4ITUbMhcvMiaNB5pI7gIPo9) (/{your_team}/RobotMotionRecord/{SongName}/) corresponding to the song.

**2.** Capture Just Dance Gameplay (Switch Screen Recording)
- Please record your Just Dance in-game screen using video capture card. We recommend using [OBS](https://obsproject.com/kb/video-capture-sources)
- Please save it in [GD](https://drive.google.com/drive/u/0/folders/1m7GYZQWQi4ITUbMhcvMiaNB5pI7gIPo9) (/{your_team}/JustDanceScreenRecord/{SongName}/) corresponding to the song.

**3.** Log In-Game Scores (Just Dance Results)
- Please write down Just Dance Score in the Google Spread sheet, [GD](https://drive.google.com/drive/u/0/folders/1m7GYZQWQi4ITUbMhcvMiaNB5pI7gIPo9) (/{your_team}/JustDanceScoreRecord_{your_team}) corresponding to the song.

**4.** Record and Save G1 Robot Trajectories

- The save file should be a comma-separated text file of 28 columns, where each row represents [timestamp(1), dof_pos(23), quat_wxyz(4)] recorded at a control frequency of 50Hz.
  <details><summary>The order of dof_pos should be in the standard order of 23DoF G1, not including the wrist dofs (click to expand):</summary>

  <pre><code>
  left_hip_pitch_joint
  left_hip_roll_joint
  left_hip_yaw_joint
  left_knee_joint
  left_ankle_pitch_joint
  left_ankle_roll_joint

  right_hip_pitch_joint
  right_hip_roll_joint
  right_hip_yaw_joint
  right_knee_joint
  right_ankle_pitch_joint
  right_ankle_roll_joint

  waist_yaw_joint
  waist_roll_joint
  waist_pitch_joint

  left_shoulder_pitch_joint
  left_shoulder_roll_joint
  left_shoulder_yaw_joint
  left_elbow_joint

  right_shoulder_pitch_joint
  right_shoulder_roll_joint
  right_shoulder_yaw_joint
  right_elbow_joint
  </code></pre>
    </details>
- Resulting text file should look like:
    ```
    1762304657.0111518,-0.20000000298023224,0.0,0.0,0.4000000059604645,-0.20000000298023224,0.0,-0.20000000298023224,0.0,0.0,0.4000000059604645,-0.20000000298023224,0.0,0.0,0.0,0.0,0.0,0.20000000298023224,0.0,1.2000000476837158,0.0,-0.20000000298023224,0.0,1.2000000476837158,0.0,0.0,0.0,1.0
    1762304657.011307,-0.20020855963230133,-4.11885412177071e-05,-0.0003218773636035621,0.39959409832954407,-0.19836343824863434,0.0007915369351394475,-0.20029042661190033,-6.851198122603819e-05,-0.00018949841614812613,0.39975854754447937,-0.1988338679075241,-0.0013329389039427042,9.838054393185303e-05,4.8836525820661336e-05,-0.0002609657822176814,-0.00035295216366648674,0.20035764575004578,-0.0021339815575629473,1.2009837627410889,6.649894203292206e-05,-0.20070603489875793,0.003482293104752898,1.2000110149383545,0.0,0.0,0.0,1.0
    1762304657.0124059,-0.20058463513851166,-0.00011414211621740833,-0.0009357244125567377,0.39882907271385193,-0.19566451013088226,0.0019640130922198296,-0.20083320140838623,-0.00020298021263442934,-0.0005314796580933034,0.3993105888366699,-0.1969119757413864,-0.0033072105143219233,0.00024817182566039264,0.0001373620907543227,-0.0007143181283026934,-0.0008566523320041597,0.2006441354751587,-0.0037887278012931347,1.202413558959961,0.00013457887689583004,-0.20138269662857056,0.00619761785492301,1.1999802589416504,1.805442661861889e-05,-0.00011980318959103897,-1.236554544448154e-05,1.0
    ...
    ```
- Please store them into [GD](https://drive.google.com/drive/u/0/folders/1m7GYZQWQi4ITUbMhcvMiaNB5pI7gIPo9) (/{your_team}/RobotTrajectoryRecord/{SongName}/) corresponding to the song.
- You can refer to the example implementation in TWIST_fork for how to: [creating csv_writer](https://github.com/whitealex95/TWIST/blob/f36ae5e82eaaf647b11426cbb5c27090531555e3/deploy_real/server_low_level_g1_sim.py#L200C1-L207C1), [logging every step](https://github.com/whitealex95/TWIST/blob/f36ae5e82eaaf647b11426cbb5c27090531555e3/deploy_real/server_low_level_g1_sim.py#L352C1-L359C1)


**Final folder should look like:**
```
/{your_team}/
│
├── RobotMotionRecord/                  # Videos of the robot dancing (front view)
│   ├── Old_Town_Road/
│   │   ├── Old_Town_Road_offline.mp4
│   │   └── Old_Town_Road_online.mp4
│   ├── Heart_Of_Glass/
│   │   ├── Heart_Of_Glass_offline.mp4
│   │   └── Heart_Of_Glass_online.mp4
│   ├── Unstoppable/
│   │   ├── Unstoppable_offline.mp4
│   │   └── Unstoppable_online.mp4
│   ├── Padam_Padam/
│   │   ├── Padam_Padam_offline.mp4
│   │   └── Padam_Padam_online.mp4
│   └── Pink_Venom/
│       ├── Pink_Venom_offline.mp4
│       └── Pink_Venom_online.mp4
│
├── RobotTrajectoryRecord/  # comma-separated txt files containing G1’s joint and quaternion trajectories
│   ├── Old_Town_Road/
│   │   ├── Old_Town_Road_offline.txt
│   │   └── Old_Town_Road_online.txt
│   ├── Heart_Of_Glass/
│   │   ├── Heart_Of_Glass_offline.txt
│   │   └── Heart_Of_Glass_online.txt
│   ├── Unstoppable/
│   │   ├── Unstoppable_offline.txt
│   │   └── Unstoppable_online.txt
│   ├── Padam_Padam/
│   │   ├── Padam_Padam_offline.txt
│   │   └── Padam_Padam_online.txt
│   └── Pink_Venom/
│       ├── Pink_Venom_offline.txt
│       └── Pink_Venom_online.txt
│
├── JustDanceScreenRecord/              # Captured Just Dance screen (via capture card)
│   ├── Old_Town_Road/
│   │   ├── Old_Town_Road_offline.mp4
│   │   └── Old_Town_Road_online.mp4
│   ├── Heart_Of_Glass/
│   │   ├── Heart_Of_Glass_offline.mp4
│   │   └── Heart_Of_Glass_online.mp4
│   ├── Unstoppable/
│   │   ├── Unstoppable_offline.mp4
│   │   └── Unstoppable_online.mp4
│   ├── Padam_Padam/
│   │   ├── Padam_Padam_offline.mp4
│   │   └── Padam_Padam_online.mp4
│   └── Pink_Venom/
│       ├── Pink_Venom_offline.mp4
│       └── Pink_Venom_online.mp4
│
└── JustDanceScoreRecord_{your_team}.xlsx   # Spreadsheet recording Just Dance scores
```

Thank you for your help with data collection!