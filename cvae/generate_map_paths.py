import numpy as np
import os
import shutil
import argparse
import rospkg
import subprocess 

def delete_mkdir(output_path):
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path, exist_ok=True)

def mkdir_if_missing(output_path):
    if os.path.exists(output_path) == False:
        os.mkdir(output_path)

class MRMHAInterface():
    
    def __init__(self, env_path_root="", env_list="", max_paths="", output_path=""):
        self.env_path_root = env_path_root
        self.env_list = env_list
        self.max_paths = max_paths
        self.output_path = output_path
        self.path_batch_num = 500
        rospack = rospkg.RosPack()
        self.PACKAGE_ROOT = rospack.get_path('walker_planner')
        self.URDF_LAUNCH_FILE = "{}/launch/planning_context_walker.launch".format(self.PACKAGE_ROOT)
        self.GEN_START_GOAL_LAUNCH_FILE = "{}/launch/generate_start_goals.launch".format(self.PACKAGE_ROOT)
        self.GEN_PATHS_LAUNCH_FILE = "{}/launch/mrmhaplanner.launch".format(self.PACKAGE_ROOT)

        # Path which will be used temporarily for generating .txt files and paths
        self.temp_path = "{}/ros_temp".format(os.getcwd())
        delete_mkdir(self.temp_path)
        os.environ["ROS_HOME"] = self.temp_path

        # Create output directory if doesnt exist, stores final paths for each envt
        mkdir_if_missing(self.output_path)

    
    def set_ros_param_from_dict(self, params):
        command = 'rosparam set / "{}"'.format(params)
        print("Running ROS Command : {}".format(command))
        subprocess.call(command, shell=True)
    
    def launch_ros_node(self, launch_file):
        command = 'roslaunch {}'.format(launch_file)
        print("Running ROS Command : {}".format(command))
        subprocess.call(command, shell=True)

    def gen_start_goals(self, env_file_path):
        '''
            Generate 500 start/goal pairs
        '''
        params = {
                "generate_start_goals/object_filename" : env_file_path
            }
        self.set_ros_param_from_dict(params)
        self.launch_ros_node(self.GEN_START_GOAL_LAUNCH_FILE)

    def gen_paths(self, env_file_path, env_output_path):
        '''
            Generate 500 paths for 500 start/goal pairs
        '''
        start_states_path = "{}/start_states.txt".format(self.temp_path)
        goal_states_path = "{}/goal_poses.txt".format(self.temp_path)
        paths_path = "{}/paths".format(self.temp_path)
        # Clear and make paths directory in ros home
        delete_mkdir(paths_path)
        params = {
                "mrmhaplanner/object_filename" : env_file_path,
                "mrmhaplanner/robot_start_states_file" : start_states_path,
                "mrmhaplanner/robot_goal_states_file" : goal_states_path
            }
        self.set_ros_param_from_dict(params)
        self.launch_ros_node(self.GEN_PATHS_LAUNCH_FILE)
        shutil.copy(start_states_path, env_output_path)
        shutil.copy(goal_states_path, env_output_path)
        shutil.move(paths_path, env_output_path)
        
    
    def run(self):
        # For each env in path and list :
            # Set environment param to environment file
            # Do Until total paths < num_paths
                # Generate 500 start goals
                # Copy generate goals to experiments folder
                # Generate 500 paths for each start/goal pair
            # Copy ~/.ros/paths/* to 
        # self.launch_ros_node(self.URDF_LAUNCH_FILE)        
        for env_i in env_list:
            env_file_path = "{}/{}.txt".format(self.env_path_root, env_i)
            
            num_batches = int(self.max_paths/self.path_batch_num)
            print("Generating for environment : {}, total batches : {}".format(env_file_path, num_batches))
            for batch_i in range(num_batches):
                print("Doing env batch : {}".format(batch_i))
                self.gen_start_goals(env_file_path)
                env_batch_output_path = "{}/{}/dump_{}".format(self.output_path, env_i, batch_i)
                # Clear and make output directory for this environment in the output_path
                delete_mkdir(env_batch_output_path)
                self.gen_paths(env_file_path, env_batch_output_path)

def parse_arguments():
    '''
        parse commandline arguments
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_path_root', required=True, dest='env_path_root', type=str, help="path where all .env files are present")
    parser.add_argument('--output_path', required=True, dest='output_path', type=str, help="path where all generated paths will be dumped")
    parser.add_argument('--env_list', required=True, nargs='+', dest='env_list', help="environment numbers to use to generate data for")
    parser.add_argument('--max_paths', required=True, dest='max_paths', type=int, help="number of paths to generate for each environment, multiples of 500")
    return parser.parse_args()

if __name__ == "__main__":
    '''
        entry point
    '''
    args = parse_arguments()
    env_path_root = args.env_path_root
    env_list = args.env_list
    output_path = args.output_path
    max_paths = args.max_paths

    mrmha = MRMHAInterface(env_path_root = env_path_root, 
                        env_list = env_list,
                        max_paths = max_paths,
                        output_path = output_path)
    mrmha.run()
    
    