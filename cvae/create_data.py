import json
import argparse
import os
import numpy as np

def label_traj(complete_sample_path, complete_env_path, complete_directory_clean, data_path_arm, data_path_base):
    '''
        get arm and base labels from a single solution_path file
    '''
    with open(data_path_base, "a") as b:
        with open(data_path_arm, "a") as a:
            with open(complete_sample_path, "r") as f:
                prev_base_joint_angles = np.zeros([1, 3])
                prev_arm_joint_angles = np.zeros([1, 7])
                
                initialized = False
                line = f.readline()
                line = f.readline()
                
                while line:
                    # import pdb
                    # pdb.set_trace()
                    
                    joint_angles = line.strip("\n").strip(" ").split(" ")
                    joint_angles = list(map((lambda x: float(x)),joint_angles))

                    base_joint_angles = np.array(joint_angles[:3])
                    arm_joint_angles = np.array(joint_angles[3:])
                    
                    if initialized:
                        prev_base_joint_angles = base_joint_angles
                        prev_arm_joint_angles = arm_joint_angles
                        initialized = True
                        continue
                    
                    epsilon = 0
                    base_moved = np.any(np.abs(base_joint_angles - prev_base_joint_angles) > epsilon)
                    arm_moved = np.any(np.abs(arm_joint_angles - prev_arm_joint_angles) > epsilon)
                    
                    if base_moved:
                        b.write("{} {}\n".format(joint_angles[0], joint_angles[1]))
                    
                    if arm_moved:
                        a.write("{} {}\n".format(joint_angles[0], joint_angles[1]))
                        
                    line = f.readline()
                    prev_base_joint_angles = base_joint_angles
                    prev_arm_joint_angles = arm_joint_angles

def parse_arguments():
    '''
        parse commandline arguments
    '''
    parser = argparse.ArgumentParser()
    # parser.add_argument('--specific', dest='specific', type=int, default=None, help="specific trajectory")
    parser.add_argument('--test', dest='test', action='store_true', help="use test data")
    parser.set_defaults(test=False)
    return parser.parse_args()

if __name__ == "__main__":
    '''
        entry point
    '''
    args = parse_arguments()
    
    if args.test:
        directory = os.path.join(os.environ["PLANNING_PROJECT_DATA"], "test")
        directory_clean = os.path.join(os.environ["PLANNING_PROJECT_DATA"], "test_clean")
    else:
        directory = os.path.join(os.environ["PLANNING_PROJECT_DATA"], "train")
        directory_clean = os.path.join(os.environ["PLANNING_PROJECT_DATA"], "train_clean")
            
    if os.path.exists(directory_clean):
        os.remove(directory_clean)
        
    os.mkdir(directory_clean)
    
    # if args.specific == None:
    env_paths = os.listdir(directory)
    
    for env_path in filter(lambda f: f[0].isdigit(), env_paths):
        os.mkdir(os.path.join(directory_clean, env_path))
        
        bulk_paths = os.listdir(os.path.join(directory, env_path))
        
        for bulk_path in filter(lambda f: f.startswith('dump'), bulk_paths):
            sample_paths = os.listdir(os.path.join(directory, env_path, bulk_path))
            
            for sample_path in filter(lambda f: f.startswith('solution_path'), sample_paths):
                complete_sample_path = os.path.join(directory, env_path, bulk_path, sample_path)
                complete_env_path = os.path.join(directory, env_path, "proj_env.env")
                complete_directory_clean = os.path.join(directory_clean, env_path)
                
                data_path_base = os.path.join(directory_clean, env_path, "data_base.txt")
                data_path_arm = os.path.join(directory_clean, env_path, "data_arm.txt")
                
                label_traj(complete_sample_path, complete_env_path, complete_directory_clean, data_path_arm, data_path_base)