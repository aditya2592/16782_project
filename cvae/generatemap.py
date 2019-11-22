import numpy as np
import os
import shutil
import argparse
import yaml

def generateGap(gap_size, start, end):
    '''
    '''
    gap_center = np.random.uniform(start+gap_size, end-gap_size)
    return [start, gap_center-gap_size/2, gap_center+gap_size/2, end]

def generateHorizontalWall(c_x, c_y, height, thickness):
    # wall left
    wall_left_center_x = (c_x[0] + c_x[1])/2
    wall_left_center_y = c_y
    wall_left_center_z = height/2
    wall_left__length = c_x[1] - c_x[0]
    wall_left__breadth = thickness
    wall_left__height = height
    
    wall_left = [wall_left_center_x, wall_left_center_y, wall_left_center_z, wall_left__length, wall_left__breadth, wall_left__height]
    
    # gap
    gap = [(c_x[1] + c_x[2])/2, c_y]
    
    # wall right
    wall_right_center_x = (c_x[3] + c_x[2])/2
    wall_right_center_y = c_y
    wall_right_center_z = height/2
    wall_right_length = c_x[3] - c_x[2]
    wall_right_breadth = thickness
    wall_right_height = height
    
    wall_right = [wall_right_center_x, wall_right_center_y, wall_right_center_z, wall_right_length, wall_right_breadth, wall_right_height]
    
    return wall_left, wall_right, gap
    
def generateVerticalWall(c_y, c_x, height, thickness):
    # wall top
    wall_top_center_x = c_x
    wall_top_center_y = (c_y[0] + c_y[1])/2
    wall_top_center_z = height/2
    wall_top__length = thickness
    wall_top__breadth = c_y[1] - c_y[0]
    wall_top__height = height
    
    wall_top = [wall_top_center_x, wall_top_center_y, wall_top_center_z, wall_top__length, wall_top__breadth, wall_top__height]
    
    # gap
    gap = [c_x, (c_y[1] + c_y[2])/2]
    
    # wall bottom
    wall_bottom_center_x = c_x
    wall_bottom_center_y = (c_y[3] + c_y[2])/2
    wall_bottom_center_z = height/2
    wall_bottom_length = thickness
    wall_bottom_breadth = c_y[3] - c_y[2]
    wall_bottom_height = height
    
    wall_bottom = [wall_bottom_center_x, wall_bottom_center_y, wall_bottom_center_z, wall_bottom_length, wall_bottom_breadth, wall_bottom_height]
    
    return wall_top, wall_bottom, gap
    
def writeWallToFile(path, wall, index):
    with open(path, "a") as f:
        f.write("wall{} {}\n".format(index, " ".join(list(map((lambda x: str(x)),wall)))))

def writeGapToFile(path, gap, index):
    with open(path, "a") as f:
        f.write("gap{} {}\n".format(index, " ".join(list(map((lambda x: str(x)),gap)))))
    
def generateRandomMap(path, config):
    '''
    '''
    length = config['map']['x_max']
    width = config['map']['y_max']
    height = config['map']['h_max']
    thickness = config['map']['wall_thickness']
    gap_size = config['map']['door_width']
    
    # hard code column 1 
    c_1_x = length/3
    
    # hard code column 2
    c_2_x = (2*length)/3
    
    # sample gap in column 1
    c_1_y = generateGap(gap_size, 0, width)
    
    # sample gap in column 2
    c_2_y = generateGap(gap_size, 0, width)
    
    # sample row 1
    r_1_y = np.random.uniform(gap_size, width-gap_size)
    
    # sample row 2
    r_2_y = np.random.uniform(gap_size, width-gap_size)
    
    # sample row 3
    r_3_y = np.random.uniform(gap_size, width-gap_size)
    
    # sample gap in row 1
    r_1_x = generateGap(gap_size, 0, c_1_x)
    
    # sample gap in row 2
    r_2_x = generateGap(gap_size, c_1_x, c_2_x)
    
    # sample gap in row 3
    r_3_x = generateGap(gap_size, c_2_x, length)
    
    # generate walls and gaps
    wall1, wall2, gap1 = generateVerticalWall(c_1_y, c_1_x, height, thickness)
    wall3, wall4, gap2 = generateVerticalWall(c_2_y, c_2_x, height, thickness)
    wall5, wall6, gap3 = generateHorizontalWall(r_1_x, r_1_y, height, thickness)
    wall7, wall8, gap4 = generateHorizontalWall(r_2_x, r_2_y, height, thickness)
    wall9, wall10, gap5 = generateHorizontalWall(r_3_x, r_3_y, height, thickness)
    
    # write gaps and walls to file
    walls = [wall1, wall2, wall3, wall4, wall5, wall6, wall7, wall8, wall9, wall10]
    gaps = [gap1, gap2, gap3, gap4, gap5]
    
    for index, wall in enumerate(walls):
        writeWallToFile(path, wall, index)
        
    for index, gap in enumerate(gaps):
        writeGapToFile(path, gap, index)

def parse_arguments():
    '''
        parse commandline arguments
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', dest='test', action='store_true', help="use test data")
    parser.add_argument('--env', dest='env', type=int, default=None, help="environment number")
    return parser.parse_args()

if __name__ == "__main__":
    '''
        entry point
    '''
    args = parse_arguments()
    
    if args.test:
        directory = os.path.join(os.environ["PLANNING_PROJECT_DATA"], "test_map")
    else:
        directory = os.path.join(os.environ["PLANNING_PROJECT_DATA"], "train_map")
    
    with open('config.yaml') as f:
        config = yaml.load(f)
    
    if os.path.exists(directory):
        shutil.rmtree(directory)
    
    os.mkdir(directory)
        
    path = os.path.join(directory,str(args.env)+".txt")
    generateRandomMap(path, config)