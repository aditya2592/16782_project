# walker_planner
Planner for Wheeled Walker.

Google Drive folder with stuff - https://drive.google.com/open?id=1duMPSww-KA-X_sNt09_XGf6uI1dzbjsR

Setup
------

1. Clone this repo (unified_planner branch) in catkin_ws/src

Also clone following :
```
git clone https://github.com/shivamvats/smpl.git -b mrmha
git clone https://github.com/aurone/leatherman
git clone https://github.com/SBPL-Cruz/wheeled_walker
```

In separate folder :
```
git clone https://github.com/shivamvats/sbpl -b mrmha
mkdir build
mkdir install
cd build
cmake -DCMAKE_INSTALL_PREFIX=../install ..
make install
```

4. Change ```~/lolocal``` in ```smpl/smpl/CmakeLists.txt``` and ```smpl/smpl/smpl-config.cmake.in``` to install path inside sbpl folder created above

5. Install also :
```
sudo apt-get install ros-kinetic-trac-ik 
sudo apt-get install libompl-dev
sudo apt-get install libgsl-dev
```
6. Build only our package: 
```catkin build walker_planner```

Rviz
------
1. Start RVIZ
2. Open the config in the repo proj.rviz
3. Once this open, map, goal states while generating goals and generate plan can be visualized

Generating Traj
--------------
Working directory : walker_planner

1. Generate map in ~/.ros/-multi_room_map.env. Copy this to env folder and rename to proj_env.env : 
```
roslaunch walker_planner generate_map.launch
cp ~/.ros/-multi_room_map.env env/proj_env.env
```

2. Generate start/goal pairs. Generates 500 start/goal pairs in ~/.ros/goal_* ~/.ros/start_*. Copy these to environments folder

```
roslaunch walker_planner generate_start_goals.launch 
cp ~/.ros/goal_* experiments
cp ~/.ros/start_states.txt experiments/
```
3. Change number of paths you need to generate plans for (start/goal pairs) in ```config/walker_right_arm.yaml``` in the ```end_planning_episode variable```. Set to 499 to generate for all start/goal pairs. **By default this is set to 0 to visualize plan for the first start/goal pair only**

4. Check that your ROS_HOME variable points to ```~/.ros```.  In your .ros run ```mkdir paths```. (This is where the solutions are stored).

5. Run planner and verify in RVIZ :
```roslaunch walker_planner mrmhaplanner.launch```

Generating Traj (Python)
------------------------
1. Launch the URDF package in sepearate shell and keep it running :
```
roslaunch walker_planner planning_context_walker.launch
```
2. Go to cvae folder
3. Run the following python script giving a list of map filname index to run the planner for :
```
python generate_map_paths.py --env_path_root $PWD/../env --output_path $PWD/test_data --env_list 2 3 4 5 6 7 8 --max_paths 2000
```
Note : The above script makes a ros_temp directory while generating stuff which is basically ROS_HOME. The final paths are copied to the path specified in --output_path 

Creating Dataset
----------------
1. Go to cvae folder
2. Run the following to generate clean dataset for env '1' only located in 'data/train'. Output is stored in 'data/train_clean'. env_path_root specifies the location where environment .env files are present. These are used for reading gap locations.
```
source activate.sh
python create_data.py --env 1 --env_path_root ../env
```
3. Run following to generate clean dataset for all env located in 'data/train'. Output is stored in 'data/train_clean'
```
source activate.sh
python create_data.py --env_path_root ../env
```
4. Run following to generate clean test dataset for all env located in 'data/test'. Output is stored in 'data/test_clean'
```
source activate.sh
python create_data.py --test --env_path_root ../env
```
5. Data will be dumped in following format in two .txt files - 'data_base.txt' and 'data_arm.txt':
```
2 (sample x,y) + 2 (start x,y) + 2 (goal x,y) + 20*2 (walls, x,y computed from x,y,z,l,b,h)
```
Training CVAE
-------------
1. Go to cvae folder.
2. Run following command to run for training base cvae. Tensorboard outputs are stored in 'experiments/cvae/{run_id}'
```
python run.py --train_dataset_root ../data/train_clean --test_dataset_root ../data/test_clean --num_epochs 250 --dataset_type arm --run_id arm_cvae
```

Testing CVAE
------------
1. Go to cvae folder
2. Run following command to load saved decoder model and run. This will plot on tensorboard and also save files in the output_path directory :
  * For arm :
  ```
  python run.py --dataset_type arm --test_only --dataset_root ../data/test_clean --decoder_path experiments/cvae/arm_walls_new_test_more_data/decoder-final.pkl --run_id test_arm --output_path arm_output --env_path_root ../env
  ```
  * For base :
  ```
  python run.py --dataset_type base --test_only --dataset_root ../data/test_clean --decoder_path experiments/cvae/base_walls_new_test_more_data/decoder-final.pkl --run_id test_b
  ase --output_path base_output
  ```
  
  File structure :
  ```
  arm_output/
   start_goal_0.txt #Start/Goal x,y for this condition 0
   gen_points_0.txt #Sampled points from ARM CVAE for this condition 0
   gen_points_fig_0.png #Plotted sampled points for this condition 0
  ```
  
  Testing Full Pipeline
  ---------------------
  ```
  python generate_map_paths.py --env_path_root $PWD/../env --output_path test_only --env_list 34 --max_paths 1 --test_only --input_path $PWD/../data/test/ --decoder_path experiments/cvae/arm_multi_env_tables/decoder-final.pkl
  ```
