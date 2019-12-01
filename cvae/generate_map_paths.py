import numpy as np
import os
import shutil
import argparse
import rospkg
import subprocess 
import matplotlib.pyplot as plt

from create_data import get_gaps
from run import CVAEInterface
from visualize_prob import generate_gaussian
from model_constants import *

def delete_mkdir(output_path):
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path, exist_ok=True)

def mkdir_if_missing(output_path):
    if os.path.exists(output_path) == False:
        os.mkdir(output_path)

class MRMHAInterface():
    
    def __init__(self, 
                env_path_root="", 
                env_list="", 
                max_paths="", 
                output_path="",
                decoder_path=""):
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

        self.decoder_path = decoder_path

    
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
        shutil.move(start_states_path, env_output_path)
        shutil.move(goal_states_path, env_output_path)
        shutil.move(paths_path, env_output_path)
        
    def run_cvae(self, env_id, gaps, start_state, goal_state):
        condition = np.hstack((start_state[:2], goal_state[:2], gaps)) 
        cvae_interface = CVAEInterface(run_id="planner_test",
                                    output_path=self.output_path,
                                    env_path_root=self.env_path_root)
        cvae_interface.load_saved_cvae(decoder_path)
        cvae_samples = cvae_interface.test_single(env_id, sample_size=1000, c_test=condition)

        fig = plt.figure()
        cvae_interface.visualize_map(env_id)
        generate_gaussian(cvae_samples, X_MAX, Y_MAX, visualize=True, fig=fig)

        return cvae_samples


    def run_test(self, input_path=""):
        '''
            Given start/goal pairs already generated, run planner
        '''
        for env_i in self.env_list:
            env_file_path = "{}/{}.txt".format(self.env_path_root, env_i)
            env_dir_path = "{}/{}/dump_0".format(input_path, env_i)
            start_states_path = "{}/start_states.txt".format(env_dir_path)
            goal_states_path = "{}/goal_poses.txt".format(env_dir_path)
            params = {
                "mrmhaplanner/object_filename" : env_file_path,
                "mrmhaplanner/robot_start_states_file" : start_states_path,
                "mrmhaplanner/robot_goal_states_file" : goal_states_path,
                "mrmhaplanner/planning/start_planning_episode" : 0,
                "mrmhaplanner/planning/end_planning_episode" : 1
            }
            self.set_ros_param_from_dict(params)
            gaps = np.array(get_gaps(env_file_path), dtype=np.float32)
            start_state = np.loadtxt(start_states_path, skiprows=1)[0,:]
            goal_state = np.loadtxt(goal_states_path, skiprows=1)[0,:]
            # Run CVAE
            cvae_samples = self.run_cvae(env_i, gaps, start_state, goal_state)
            # Do GMM

            # Run Planner
            # self.launch_ros_node(self.GEN_PATHS_LAUNCH_FILE)


    def run_generate(self):
        '''
            Generate start/goal pairse and corresponding paths. Used for generating training data
        '''
        # For each env in path and list :
            # Set environment param to environment file
            # Do Until total paths < num_paths
                # Generate 500 start goals
                # Copy generate goals to experiments folder
                # Generate 500 paths for each start/goal pair
            # Copy ~/.ros/paths/* to 
        # self.launch_ros_node(self.URDF_LAUNCH_FILE)        
        for env_i in self.env_list:
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
    parser.add_argument('--max_paths', required=True, dest='max_paths', type=int, help="number of paths to generate for each environment, multiples of 500, or number of paths in each envt to test on in test only mode")
    
    parser.add_argument('--test_only', dest='test_only', action='store_true', help="Whether to run planner in test only mode on start/goal pairs existing")
    parser.add_argument('--input_path', required=False, dest='input_path', type=str, help="path where all start/goal pairs are present for envts", default="")
    parser.add_argument('--decoder_path', dest='decoder_path', type=str, help='path to decoder model for testing', default="")
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
    test_only = args.test_only
    input_path = args.input_path
    decoder_path = args.decoder_path

    mrmha = MRMHAInterface(env_path_root = env_path_root, 
                        env_list = env_list,
                        max_paths = max_paths,
                        output_path = output_path,
                        decoder_path = decoder_path)
    if test_only:
        print("Running test only mode")
        assert(input_path != "")
        assert(decoder_path != "")
        mrmha.run_test(input_path=input_path)
    else:
        mrmha.run_generate()
    
    