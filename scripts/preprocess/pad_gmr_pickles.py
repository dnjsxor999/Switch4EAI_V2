import pickle
import numpy as np

# Dictionary mapping filename -> frames to pad at the beginning (repeating the first frame)
start_pads = {
    "Heart_Of_Glass": 238,
    "Old_Town_Road": 123,
    "Padam_Padam": 735,
    "Unstoppable": 365,
    "Pink_Venom": 450,
}

# Update to your directory containing the pickles
pickle_dir = "YOUR_PICKLE_DIRECTORY_HERE"

for song, pad in start_pads.items():
    # output file name -> same base, but with _padded.mp4 suffix
    infile_path = f"{pickle_dir}/{song}_cut/{song}_cut_poses.pkl"
    outfile_name = f"{song}_cut_poses_padded.pkl"
    outfile_path = f"{infile_path.rsplit('/', 1)[0]}/{outfile_name}"
    print(f"Padding {infile_path} -> {outfile_path} ({pad} frames)")

    pose_data = np.load(infile_path, allow_pickle=True)
    pose_data['fps'] = pose_data['fps']
    # Pad at the start with the first frame (mode='edge' repeats edge values)
    pose_data['root_pos'] = np.pad(pose_data['root_pos'], ((pad, 0), (0, 0)), mode='edge')
    pose_data['root_rot'] = np.pad(pose_data['root_rot'], ((pad, 0), (0, 0)), mode='edge')
    pose_data['dof_pos'] = np.pad(pose_data['dof_pos'], ((pad, 0), (0, 0)), mode='edge')
    pose_data['local_body_pos'] = np.pad(pose_data['local_body_pos'], ((pad, 0), (0, 0), (0, 0)), mode='edge') if pose_data['local_body_pos'] is not None else None
    pose_data['link_body_list'] = pose_data['link_body_list']
    
    with open(outfile_path, "wb") as f:
        pickle.dump(pose_data, f)

print("All pads completed!")
