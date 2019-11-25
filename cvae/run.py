import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
import os
import csv
from random import randint, random
import time
import argparse
from torch.utils.data import DataLoader
import torch.nn.functional as F
import shutil

from model import CVAE
from model_constants import *
from paths_dataset import PathsDataset


class CVAEInterface():
    def __init__(self, run_id=1, output_path=""):
        super().__init__()
        self.cvae = CVAE(run_id=run_id)
        self.device = torch.device('cuda' if CUDA_AVAILABLE else 'cpu')
        self.output_path = output_path

        if self.output_path is not None:
            if os.path.exists(self.output_path):
                shutil.rmtree(self.output_path)
            os.mkdir(self.output_path)
    


    def load_dataset(self, dataset_root, data_type="arm", mode="train"):
        assert(data_type == "both" or data_type == "arm" or data_type == "base")
        assert(mode == "train" or mode == "test")
        # Should show different count and path for different modes
        print("Loading {} dataset for mode : {}, path : {}".format(data_type, mode, dataset_root))
        self.data_type = data_type

        paths_dataset = PathsDataset(type="FULL_STATE")
        c_test_dataset = PathsDataset(type="CONDITION_ONLY")
        env_dir_paths = os.listdir(dataset_root)
        # Get all C vars to test sample generation on each
        all_condition_vars = []
        for env_dir_index in filter(lambda f: f[0].isdigit(), env_dir_paths):
            env_paths_file = os.path.join(dataset_root, env_dir_index, "data_{}.txt".format(data_type))
            env_paths = np.loadtxt(env_paths_file)
            all_condition_vars += env_paths[:,X_DIM:X_DIM+C_DIM].tolist()
            # print(env_paths.shape)
            # Take only required elements
            env_paths = env_paths[:, :X_DIM + C_DIM]
            # Uniquify to remove duplicates
            env_paths = np.unique(env_paths, axis=0)
            env_index = np.empty((env_paths.shape[0], 1))
            env_index.fill(env_dir_index)
            data = np.hstack((env_index, env_paths))
            paths_dataset.add_env_paths(data.tolist())
            print("Added {} states from {} environment".format(env_paths.shape[0], env_dir_index))

        dataloader = DataLoader(paths_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)

        if data_type != "both":
            self.all_condition_vars = np.unique(all_condition_vars, axis=0)
            print("Unique test conditions count : {}".format(self.all_condition_vars.shape[0]))
            all_condition_vars_tile = np.repeat(self.all_condition_vars, TEST_SAMPLES, 0)
            c_test_dataset.add_env_paths(all_condition_vars_tile.tolist())
            c_test_dataloader = DataLoader(c_test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=False)

            # Depending on which dataset is being loaded, set the right variables
            if mode == "train":
                self.train_dataloader = dataloader
            elif mode == "test":
                self.test_dataloader = c_test_dataloader 

        else:
            arm_test_dataset = PathsDataset(type="CONDITION_ONLY")
            base_test_dataset = PathsDataset(type="CONDITION_ONLY")

            all_condition_vars = np.array(all_condition_vars)
            self.all_condition_vars = np.delete(all_condition_vars, [4, 5], axis=1)
            self.all_condition_vars = np.unique(self.all_condition_vars, axis=0)
            print("Unique test conditions count : {}".format(self.all_condition_vars.shape[0]))
            # print(self.all_condition_vars)
            arm_condition_vars = np.insert(self.all_condition_vars, 2*POINT_DIM, 1, axis=1)
            arm_condition_vars = np.insert(arm_condition_vars, 2*POINT_DIM, 0, axis=1)

            arm_condition_vars = np.repeat(arm_condition_vars, TEST_SAMPLES, 0)
            arm_test_dataset.add_env_paths(arm_condition_vars.tolist())
            arm_test_dataloader = DataLoader(arm_test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=False)

            base_condition_vars = np.insert(self.all_condition_vars, 2*POINT_DIM, 0, axis=1)
            base_condition_vars = np.insert(base_condition_vars, 2*POINT_DIM, 1, axis=1)

            base_condition_vars = np.repeat(base_condition_vars, TEST_SAMPLES, 0)
            base_test_dataset.add_env_paths(base_condition_vars.tolist())
            base_test_dataloader = DataLoader(base_test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=False)

            if mode == "train":
                self.train_dataloader = dataloader
            elif mode == "test":
                self.arm_test_dataloader = arm_test_dataloader
                self.base_test_dataloader = base_test_dataloader



    def plot(self, x, c, walls=False, suffix=0, write_file=False):
        # print(c)
        start = c[0:2]
        goal = c[2:4]
        # For given conditional, plot the samples
        fig1 = plt.figure(figsize=(10, 6), dpi=80)
        ax1 = fig1.add_subplot(111, aspect='equal')
        plt.scatter(x[:, 0], x[:, 1], color="green", s=70, alpha=0.1)
        plt.scatter(start[0], start[1], color="blue", s=70, alpha=0.6)
        plt.scatter(goal[0], goal[1], color="red", s=70, alpha=0.6)
        if walls:
            wall_locs = c[4:]
            i = 0
            while i < wall_locs.shape[0]:
                plt.scatter(wall_locs[i], wall_locs[i+1], color="yellow", s=70, alpha=0.6)
                i = i + 2

        plt.xlim(0, X_MAX)
        plt.ylim(0, Y_MAX)
        if write_file:
            plt.savefig('{}/gen_points_fig_{}.png'.format(self.output_path, suffix))
            np.savetxt('{}/gen_points_{}.txt'.format(self.output_path, suffix), x, fmt="%.2f", delimiter=',')
            np.savetxt('{}/start_goal_{}.txt'.format(self.output_path, suffix), np.vstack((start, goal)), fmt="%.2f", delimiter=',')
        # plt.show()
        # plt.close(fig1)
        return fig1

    def load_saved_cvae(self, decoder_path):
        self.cvae.load_decoder(decoder_path)

        # base_cvae = CVAE(run_id=run_id)
        # base_decoder_path = 'experiments/cvae/base/decoder-final.pkl'
        # base_cvae.load_decoder(base_decoder_path)

        # for iteration, batch in enumerate(dataloader):

    def test(self, epoch, dataloader, write_file=False, suffix=""):

        x_test_predicted = []
        self.cvae.eval()
        for iteration, batch in enumerate(dataloader):
            # print(batch)
            c_test_data = batch.float().to(self.device)
            # print(c_test_data[0,:])
            x_test = self.cvae.batch_inference(c=c_test_data)
            x_test_predicted += x_test.detach().cpu().numpy().tolist()
            # print(x_test.shape)
            if iteration % LOG_INTERVAL == 0 or iteration == len(dataloader)-1:
                print("Test Epoch {:02d}/{:02d} Batch {:04d}/{:d}, Iteration {}".format(
                    epoch, num_epochs, iteration, len(dataloader)-1, iteration))

        x_test_predicted = np.array(x_test_predicted)
        # print(x_test_predicted.shape)
        # Draw plot for each unique condition
        for c_i in range(self.all_condition_vars.shape[0]):
            x_test = x_test_predicted[c_i * TEST_SAMPLES : (c_i + 1) * TEST_SAMPLES]
            # Fine because c_test is used only for plotting, we dont need arm/base label here
            c_test = self.all_condition_vars[c_i, :]
            fig = self.plot(x_test, c_test, suffix=c_i, write_file=write_file)
            self.cvae.tboard.add_figure('test_epoch_{}/condition_{}_{}'.format(epoch, c_i, suffix), fig, 0)
            if c_i % LOG_INTERVAL == 0:
                print("Plotting condition : {}".format(c_i))
        self.cvae.tboard.flush()

        # for c_i in range(c_test_data.shape[0]):
        #     c_test = c_test_data[c_i,:]
        #     c_test_gpu = torch.from_numpy(c_test).float().to(device)
            
        #     x_test = cvae_model.inference(n=TEST_SAMPLES, c=c_test_gpu)
        #     x_test = x_test.detach().cpu().numpy()
        #     fig = plot(x_test, c_test)
        #     cvae_model.tboard.add_figure('test_epoch_{}/condition_{}'.format(epoch, c_i), fig, 0)

        #     if c_i % 50 == 0:
        #         print("Epoch : {}, Testing condition count : {} ".format(epoch, c_i))


    def train(self,
            run_id=1,
            num_epochs=1,
            initial_learning_rate=0.001,
            weight_decay=0.0001):
        
        optimizer = torch.optim.Adam(self.cvae.parameters(), lr=initial_learning_rate, weight_decay=weight_decay)
        for epoch in range(num_epochs):
            for iteration, batch in enumerate(self.train_dataloader):
                # print(batch['condition'][0,:])
                self.cvae.train()
                x = batch['state'].float().to(self.device)
                c = batch['condition'].float().to(self.device)
                recon_x, mean, log_var, z = self.cvae(x, c)
                # print(recon_x.shape)

                loss = self.cvae.loss_fn(recon_x, x, mean, log_var)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                counter = epoch * len(self.train_dataloader) + iteration
                if iteration % LOG_INTERVAL == 0 or iteration == len(self.train_dataloader)-1:
                    print("Train Epoch {:02d}/{:02d} Batch {:04d}/{:d}, Iteration {}, Loss {:9.4f}".format(
                        epoch, num_epochs, iteration, len(self.train_dataloader)-1, counter, loss.item()))
                    self.cvae.tboard.add_scalar('train/loss', loss.item(), counter)

                    # cvae.eval()
                    # c_test = c[0,:]
                    # x_test = cvae.inference(n=TEST_SAMPLES, c=c_test)
                    # x_test = x_test.detach().cpu().numpy()
                    # fig = plot(x_test, c_test)
                    # cvae.tboard.add_figure('test/samples', fig, counter)

            if epoch % TEST_INTERVAL == 0 or epoch == num_epochs - 1:
                # Test CVAE for all c by drawing samples
                if self.data_type != "both":
                    self.test(epoch, self.test_dataloader)
                else:
                    self.test(epoch, self.arm_test_dataloader, suffix="arm")
                    self.test(epoch, self.base_test_dataloader, suffix="base")

                
            if epoch % SAVE_INTERVAL == 0 and epoch > 0:
                self.cvae.save_model_weights(counter)

        self.cvae.save_model_weights('final')
    

def parse_arguments():
    # Command-line flags are defined here.
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_id', dest='run_id', type=str, default=1)
    parser.add_argument('--num_epochs', dest='num_epochs', type=int, default=10)
    parser.add_argument('--train_dataset_root', dest='train_dataset_root', type=str, required=True)
    parser.add_argument('--test_dataset_root', dest='test_dataset_root', type=str, required=True)
    parser.add_argument('--dataset_type', dest='dataset_type', type=str, help='choose arm/base', default='arm')
    parser.add_argument('--test_only', dest='test_only', action='store_true', help="Whether to use saved model and run test only")
    parser.add_argument('--decoder_path', dest='decoder_path', type=str, help='path to decoder model', default=None)
    parser.add_argument('--output_path', dest='output_path', type=str, help='path to store output plots and files in test mode', default=None)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    run_id = args.run_id
    num_epochs = args.num_epochs
    train_dataset_root = args.train_dataset_root
    test_dataset_root = args.test_dataset_root
    dataset_type = args.dataset_type
    test_only = args.test_only
    decoder_path = args.decoder_path
    output_path = args.output_path

        

    print("CUDA_AVAILABLE : {}".format(CUDA_AVAILABLE))
    cvae_interface = CVAEInterface(run_id=run_id,
                                    output_path=output_path)

    cvae_interface.load_dataset(train_dataset_root, data_type=dataset_type, mode="train")
    cvae_interface.load_dataset(test_dataset_root, data_type=dataset_type, mode="test")
    if test_only:
        if decoder_path is None or output_path is None:
            raise Exception("All inputs not provided for test mode")
        cvae_interface.load_saved_cvae(decoder_path)
        cvae_interface.test(0, cvae_interface.test_dataloader, write_file=True)
    else:
        cvae_interface.train(
                    run_id=run_id,
                    num_epochs=num_epochs,
                    initial_learning_rate=INITIAL_LEARNING_RATE,
                    weight_decay=WEIGHT_DECAY)


# (restrict tensorflow memory growth)
# os.environ["CUDA_VISIBLE_DEVICES"]="1"
# config = tf.ConfigProto()
# config.gpu_options.allow_growth=True

# # neural network parameters
# mb_size = 256
# h_Q_dim = 512
# h_P_dim = 512

# c = 0
# lr = 1e-4

# # problem dimensions
# dim = 6
# dataElements = dim+3*3+2*dim # sample (6D), gap1 (2D, 1D orientation), gap2, gap3, init (6D), goal (6D)

# z_dim = 3 # latent
# X_dim = dim # samples
# y_dim = dim # reconstruction of the original point
# c_dim = dataElements - dim # dimension of conditioning variable

# # read in data from csv
# filename = 'narrowDataFile.txt'
# f = open(filename, 'rb')
# reader = csv.reader(f, delimiter=',')
# count = 0
# data_list = []
# for row in reader:
#     data_list.append(map(float,row[0:dataElements]))

# data = np.array(data_list,dtype='d')
# numEntries = data.shape[0]


# # split the inputs and conditions into test train (to be processed in the next step into an occupancy grid representation)
# ratioTestTrain = 0.8;
# numTrain = int(numEntries*ratioTestTrain)

# X_train = data[0:numTrain,0:dim] # state: x, y, z, xdot, ydot, zdot
# c_train = data[0:numTrain,dim:dataElements] # conditions: gaps, init (6), goal (6)

# X_test = data[numTrain:numEntries,0:dim]
# c_test = data[numTrain:numEntries,dim:dataElements]
# numTest = X_test.shape[0]

#  change conditions to occupancy grid
# def isSampleFree(sample, obs):
#     for o in range(0,obs.shape[0]/(2*dimW)):
#         isFree = 0
#         for d in range(0,sample.shape[0]):
#             if (sample[d] < obs[2*dimW*o + d] or sample[d] > obs[2*dimW*o + d + dimW]):
#                 isFree = 1
#                 break
#         if isFree == 0:
#             return 0
#     return 1

# gridSize = 11
# dimW = 3
# plotOn = False;

# # process data into occupancy grid
# conditions = data[0:numEntries,dim:dataElements]
# conditionsOcc = np.zeros([numEntries,gridSize*gridSize])
# occGridSamples = np.zeros([gridSize*gridSize, 2])
# gridPointsRange = np.linspace(0,1,num=gridSize)

# idx = 0;
# for i in gridPointsRange:
#     for j in gridPointsRange:
#         occGridSamples[idx,0] = i
#         occGridSamples[idx,1] = j
#         idx += 1;

# start = time.time();
# for j in range(0,numEntries,1):
#     dw = 0.1
#     dimW = 3
#     gap1 = conditions[j,0:3]
#     gap2 = conditions[j,3:6]
#     gap3 = conditions[j,6:9]
#     init = conditions[j,9:15]
#     goal = conditions[j,15:21]

#     obs1 = [0, gap1[1]-dw, -0.5,             gap1[0], gap1[1], 1.5]
#     obs2 = [gap2[0]-dw, 0, -0.5,             gap2[0], gap2[1], 1.5];
#     obs3 = [gap2[0]-dw, gap2[1]+dw, -0.5,    gap2[0], 1, 1.5];
#     obs4 = [gap1[0]+dw, gap1[1]-dw, -0.5,    gap3[0], gap1[1], 1.5];
#     obs5 = [gap3[0]+dw, gap1[1]-dw, -0.5,    1, gap1[1], 1.5];
#     obs = np.concatenate((obs1, obs2, obs3, obs4, obs5), axis=0)
    
#     if j % 5000 == 0:
#         print('Iter: {}'.format(j))
        
#     occGrid = np.zeros(gridSize*gridSize)
#     for i in range(0,gridSize*gridSize):
#         occGrid[i] = isSampleFree(occGridSamples[i,:],obs)
#     conditionsOcc[j,:] = occGrid
    
#     if plotOn:
#         fig1 = plt.figure(figsize=(10,6), dpi=80)
#         ax1 = fig1.add_subplot(111, aspect='equal')
#         for i in range(0,obs.shape[0]/(2*dimW)): # plot obstacle patches
#             ax1.add_patch(
#             patches.Rectangle(
#                 (obs[i*2*dimW], obs[i*2*dimW+1]),   # (x,y)
#                 obs[i*2*dimW+dimW] - obs[i*2*dimW],          # width
#                 obs[i*2*dimW+dimW+1] - obs[i*2*dimW+1],          # height
#                 alpha=0.6
#             ))
#         for i in range(0,gridSize*gridSize): # plot occupancy grid
#             if occGrid[i] == 0:
#                 plt.scatter(occGridSamples[i,0], occGridSamples[i,1], color="red", s=70, alpha=0.8)
#             else:
#                 plt.scatter(occGridSamples[i,0], occGridSamples[i,1], color="green", s=70, alpha=0.8)
#         plt.show()
# end = time.time();
# print('Time: ', end-start)
    
# cs = np.concatenate((data[0:numEntries,dim+3*dimW:dataElements], conditionsOcc), axis=1) # occ, init, goal
# c_dim = cs.shape[1]
# c_gapsInitGoal = c_test
# c_train = cs[0:numTrain,:] 
# c_test = cs[numTrain:numEntries,:]


#  define networks
# X = tf.placeholder(tf.float32, shape=[None, X_dim])
# c = tf.placeholder(tf.float32, shape=[None, c_dim])

# # Q
# inputs_Q = tf.concat(axis=1, values=[X,c])

# dense_Q1 = tf.layers.dense(inputs=inputs_Q, units=h_Q_dim, activation=tf.nn.relu)
# dropout_Q1 = tf.layers.dropout(inputs=dense_Q1, rate=0.5)
# dense_Q2 = tf.layers.dense(inputs=dropout_Q1, units=h_Q_dim, activation=tf.nn.relu)

# z_mu = tf.layers.dense(inputs=dense_Q2, units=z_dim) # output here is z_mu
# z_logvar = tf.layers.dense(inputs=dense_Q2, units=z_dim) # output here is z_logvar

# # P
# eps = tf.random_normal(shape=tf.shape(z_mu))
# z = z_mu + tf.exp(z_logvar / 2) * eps
# inputs_P = tf.concat(axis=1, values=[z,c])

# dense_P1 = tf.layers.dense(inputs=inputs_P, units=h_P_dim, activation=tf.nn.relu)
# dropout_P1 = tf.layers.dropout(inputs=dense_P1, rate=0.5)
# dense_P2 = tf.layers.dense(inputs=dropout_P1, units=h_P_dim, activation=tf.nn.relu)

# y = tf.layers.dense(inputs=dense_P2, units=X_dim) # fix to also output y

# # training
# w = [[1, 1, 1, 0.5, 0.5, 0.5]];
# recon_loss = tf.losses.mean_squared_error(labels=X, predictions=y, weights=w)
# # TODO: fix loss function for angles going around
# kl_loss = 10**-4 * 2 * tf.reduce_sum(tf.exp(z_logvar) + z_mu**2 - 1. - z_logvar, 1)

# cvae_loss = tf.reduce_mean(kl_loss + recon_loss)

# train_step = tf.train.AdamOptimizer(lr).minimize(cvae_loss)

# sess = tf.Session(config=config)
# sess.run(tf.global_variables_initializer())
# it = 0;

# for it in range(it,it+500001):
#     # randomly generate batches
#     batch_elements = [randint(0,numTrain-1) for n in range(0,mb_size)]
#     X_mb = X_train[batch_elements,:]
#     c_mb = c_train[batch_elements,:]

#     _, loss = sess.run([train_step, cvae_loss], feed_dict={X: X_mb, c: c_mb})

#     if it % 1000 == 0:
#         print('Iter: {}'.format(it))
#         print('Loss: {:.4}'. format(loss))
#         print()
        
# # plot the latent space
# num_viz = 3000

# vizIdx = randint(0,numTest-1);
# print vizIdx
# c_sample_seed = c_test[vizIdx,:]
# c_sample = np.repeat([c_sample_seed],num_viz,axis=0)
# c_viz = c_gapsInitGoal[vizIdx,:]

# # directly sample from the latent space (preferred, what we will use in the end)
# y_viz, z_viz = sess.run([y, z], feed_dict={z: np.random.randn(num_viz, z_dim), c: c_sample})

# fig1 = plt.figure(figsize=(10,6), dpi=80)
# ax1 = fig1.add_subplot(111, aspect='equal')

# plt.scatter(y_viz[:,0],y_viz[:,1], color="green", s=70, alpha=0.1)

# dw = 0.1
# dimW = 3
# gap1 = c_viz[0:3]
# gap2 = c_viz[3:6]
# gap3 = c_viz[6:9]
# init = c_viz[9:15]
# goal = c_viz[15:21]

# obs1 = [0, gap1[1]-dw, -0.5,             gap1[0], gap1[1], 1.5]
# obs2 = [gap2[0]-dw, 0, -0.5,             gap2[0], gap2[1], 1.5];
# obs3 = [gap2[0]-dw, gap2[1]+dw, -0.5,    gap2[0], 1, 1.5];
# obs4 = [gap1[0]+dw, gap1[1]-dw, -0.5,    gap3[0], gap1[1], 1.5];
# obs5 = [gap3[0]+dw, gap1[1]-dw, -0.5,    1, gap1[1], 1.5];
# obsBounds = [-0.1, -0.1, -0.5, 0, 1.1, 1.5,
#             -0.1, -0.1, -0.5, 1.1, 0, 1.5,
#             -0.1, 1, -0.5, 1.1, 1.1, 1.5,
#             1, -0.1, -0.5, 1.1, 1.1, 1.5,]

# obs = np.concatenate((obs1, obs2, obs3, obs4, obs5, obsBounds), axis=0)
# for i in range(0,obs.shape[0]/(2*dimW)):
#     ax1.add_patch(
#     patches.Rectangle(
#         (obs[i*2*dimW], obs[i*2*dimW+1]),   # (x,y)
#         obs[i*2*dimW+dimW] - obs[i*2*dimW],          # width
#         obs[i*2*dimW+dimW+1] - obs[i*2*dimW+1],          # height
#         alpha=0.6
#     ))
    
# for i in range(0,gridSize*gridSize): # plot occupancy grid
#     cIdx = i + 2*dim
#     if c_sample_seed[cIdx] == 0:
#         plt.scatter(occGridSamples[i,0], occGridSamples[i,1], color="red", s=50, alpha=0.7)
#     else:
#         plt.scatter(occGridSamples[i,0], occGridSamples[i,1], color="green", s=50, alpha=0.7)

# plt.scatter(init[0], init[1], color="red", s=250, edgecolors='black') # init
# plt.scatter(goal[0], goal[1], color="blue", s=250, edgecolors='black') # goal

# plt.show()

# plt.figure(figsize=(10,6), dpi=80)
# viz1 = 1;
# viz2 = 4;
# plt.scatter(y_viz[:,viz1],y_viz[:,viz2], color="green", s=70, alpha=0.1)
# plt.scatter(c_viz[viz1+9],c_viz[viz2+9], color="red", s=250, edgecolors='black') # init
# plt.scatter(c_viz[viz1+9+dim],c_viz[viz2+9+dim], color="blue", s=500, edgecolors='black') # goal
# plt.show()