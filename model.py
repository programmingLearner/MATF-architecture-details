# Tianyang Zhao
# Architecture Details for CVPR 19 paper: Multi-Agent Tensor Fusion for Contextual Trajectory Prediction
# Link: http://openaccess.thecvf.com/content_CVPR_2019/html/Zhao_Multi-Agent_Tensor_Fusion_for_Contextual_Trajectory_Prediction_CVPR_2019_paper.html
# ArXiv Link: https://arxiv.org/abs/1904.04776
# Feel free to contact: tyzhao@ucla.edu; Please include 'MATF' in the email title  

## Code for MATF main architecture

# Imports
import torch
import torch.nn as nn
import torchvision
import numpy as np
import torch.nn.functional as F
from utils import weights_init, conv2DBatchNormRelu, conv2DRelu, deconv2DBatchNormRelu, deconv2DRelu
import torchvision.models as models
import matplotlib.pyplot as plt


###########################################################################################
##                                                                                       ##
##                                        helpers                                        ##
##                                                                                       ##
###########################################################################################

class SemanticImageEncoder(nn.Module):
    '''
    Tianyang:
    Simple Convolutional Encoder for semantic map images
    Input size: flexible
    Output size: 1/2 input
    '''

    def __init__(self, in_channels = 3, out_channels = 32):
        super(SemanticImageEncoder, self).__init__()

        self._encoder = nn.Sequential(
            conv2DBatchNormRelu(in_channels = in_channels, n_filters = 16, \
                k_size = 3,  stride = 1, padding = 1),
            conv2DBatchNormRelu(in_channels = 16, n_filters = 16, \
                k_size = 4,  stride = 1, padding = 2),
            nn.MaxPool2d((2, 2), stride=(2, 2)),

            conv2DBatchNormRelu(in_channels = 16, n_filters = out_channels, \
                k_size = 5,  stride = 1, padding = 2),
            )

    def forward(self, input):
        encoded = self._encoder(input.type(torch.cuda.FloatTensor))
        return encoded


class resnetShallow(nn.Module):
	'''
    Tianyang:
    ResNet Encoder for semantic map image
    Output Size: 30 * 30
    '''
    def __init__(self):
        super(resnetShallow, self).__init__()

        self.trunk = torchvision.models.resnet18(pretrained=True)

        self.upscale3 = nn.Sequential(
                nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),)

        self.upscale4 = nn.Sequential(
                nn.ConvTranspose2d(512, 128, 7, stride=4, padding=3, output_padding=3),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),)

        self.shrink = conv2DBatchNormRelu(in_channels = 384, n_filters = 32, \
                k_size = 1,  stride = 1, padding = 0)

    def forward(self, input):
        x = self.trunk.conv1(input.type(torch.cuda.FloatTensor))
        x = self.trunk.bn1(x)
        x = self.trunk.relu(x)
        x = self.trunk.maxpool(x)

        x = self.trunk.layer1(x)
        x2 = self.trunk.layer2(x)  # /8 the size
        x3 = self.trunk.layer3(x2) # 16
        x4 = self.trunk.layer4(x3) # 32

        x3u = self.upscale3(x3.detach())
        x4u = self.upscale4(x4.detach())

        xall = torch.cat((x2.detach(), x3u, x4u), dim=1)
        xall = F.upsample(xall, size=(30,30))
        output = self.shrink(xall)

        return output


class SpatialPoolAgent(nn.Module):
    '''
    Tianyang:
    Spatially place and max pool encoded agents states to the spatial grid of the same shape (h * w)
        to the emncoded semantic image feature map w.r.t. their corresponding pooled coordinates
    Note that the pooling layer have no parameters to train, 
    but should be called multiple times (= number of agents) iteratively 
        to pool every agent info, 
    the order of the calling sequence will not lead to different results
    '''
    def __init__(self):
        super(SpatialPoolAgent, self).__init__()

    def forward(self, input_grid, input_state, coordinate, batch_idx):
        '''
        Params: input_grid: spatial grid input of shape (batch * c * h * w), 
                            this is independent of the semantic image,
                            this is supposed to be the placed and pooled map of agents so far 
                            (since this function is iteratively called);
                            0 as init.
                            c is supposed to be identical to the dimension of the states
                                of 'input states',
                            h * w is supposed to be identical as those of the semantic image,
                            the final output of this placing and pooling Module will be concated with
                                the semantic image in the channel dimension
                input_state: input state vector (output from agent encoder) of shape
                            (batch (must = 1) * c * 1)
                coordinate: input coordinate of shape (2)
                batch index: int, for scene, not for agent: e.g. 3 scenes with agents (7,5,9), then bi < 3
        Return: placed and pooled map of agents
        '''

        bi = batch_idx
        ori_state = input_grid[bi, :, coordinate[0], coordinate[1]]
        pooled_state = torch.max(ori_state.type(torch.cuda.FloatTensor), input_state[0, :, 0].type(torch.cuda.FloatTensor))

        input_grid[bi, :,coordinate[0], coordinate[1]] = pooled_state
        return input_grid


class SpatialFetchAgent(nn.Module):
    '''
    Tianyang:
    Spatially fetch back fused agents states from fused scene and return the sum of (residual) them and
        the agent's original encoded states
    '''
    def __init__(self, encoding_dim = 32):
        super(SpatialFetchAgent, self).__init__()
        self._encoding_dim = encoding_dim

    def forward(self, fused_scene, individual_state, coordinate, batch_idx, pretrain = False):
        # pretrain == True for pre-training, no residual from fusion
        # for details of other params, refer to SpatialPoolAgent

        bi = batch_idx
            # bi: batch index

        if pretrain:
            output = individual_state[0, :, :] 
        else:
            fused_state = fused_scene[bi, :, coordinate[0], coordinate[1]]
            fused_state_dim2 = fused_state.view(self._encoding_dim, 1)
            output = fused_state_dim2 + individual_state[0, :, :]                       

        return output


class AgentsMapFusion(nn.Module):
    '''
    Tianyang:
    Concat encoded agents grid and encoded semantic image,
        then do fully convolution to infer about 
        scene context and social interaction
    '''
    def __init__(self, in_channels = 32 + 32, out_channels = 32):
        super(AgentsMapFusion, self).__init__()

        self._conv1 = conv2DBatchNormRelu(in_channels = in_channels, n_filters = out_channels, \
                k_size = 3,  stride = 1, padding = 1, dilation = 1)
        self._pool1 = nn.MaxPool2d((2, 2), stride=(2, 2))
        self._conv2 = conv2DBatchNormRelu(in_channels = out_channels, n_filters = out_channels, \
                k_size = 3,  stride = 1, padding = 1, dilation = 1)
        self._pool2 = nn.MaxPool2d((2, 2), stride=(2, 2))
        self._conv3 = conv2DBatchNormRelu(in_channels = out_channels, n_filters = out_channels, \
                k_size = 4,  stride = 1, padding = 1, dilation = 1)

        self._deconv2 = deconv2DBatchNormRelu(in_channels = out_channels, n_filters = out_channels, \
                k_size = 4, stride = 2, padding = 1)


    def forward(self, input_agent, input_map):
        cat = torch.cat((input_map.type(torch.cuda.FloatTensor), input_agent.type(torch.cuda.FloatTensor)), 1)
        
        conv1 = self._conv1.forward(cat)
        conv2 = self._conv2.forward(self._pool1.forward(conv1))
        conv3 = self._conv3.forward(self._pool2.forward(conv2))

        up2 = self._deconv2.forward(conv2)
        up3 = F.upsample(conv3, scale_factor=5)

        features = conv1 + up2 + up3
        return features


class AgentEncoderLSTM(nn.Module):
    '''
    This part of the code is revised from Social GAN's paper for fair comparison
    Link to their original code: https://github.com/agrimgupta92/sgan
    [Social GAN: Socially Acceptable Trajectories with Generative Adversarial Networks 
	 Agrim Gupta, Justin Johnson, Fei-Fei Li, Silvio Savarese, Alexandre Alahi 
	 Presented at CVPR 2018]
    run on all the agents individually: batch_idx
    '''
    def __init__(self, input_dim = 2, embedding_dim = 32, h_dim = 32, mlp_dim = 512, num_layers = 1, dropout = 0.3):
        super(AgentEncoderLSTM, self).__init__()

        self._mlp_dim = mlp_dim
        self._h_dim = h_dim
        self._embedding_dim = embedding_dim
        self._num_layers = num_layers
        self._input_dim = input_dim

        self._encoder = nn.LSTM(embedding_dim, h_dim, num_layers, dropout=dropout)
        self._spatial_embedding = nn.Linear(input_dim, embedding_dim)

    def init_hidden(self, batch_size):
        # batch size should be number of agents in the whole batch, instead of number of scenes
        return (
            torch.zeros(self._num_layers, batch_size, self._h_dim).cuda(),
            torch.zeros(self._num_layers, batch_size, self._h_dim).cuda()
        )

    def forward(self, obs_traj):
        '''
        Inputs:
        - obs_traj: Tensor of shape (obs_len, batch size, 2)
        Output:
        - final_h: Tensor of shape (self.num_layers, batch size, self.h_dim)
        '''
        # convert to relative, as Social GAN do
        rel_curr_ped_seq = torch.tensor(np.zeros(obs_traj.shape)).type(torch.cuda.FloatTensor)
        rel_curr_ped_seq[1:,:,:] = obs_traj[1:,:,:] - obs_traj[:-1,:,:]

        # Encode observed Trajectory
        batch = obs_traj.size(1)
        obs_traj_embedding = self._spatial_embedding(rel_curr_ped_seq.view(-1, self._input_dim))
        obs_traj_embedding = obs_traj_embedding.view(-1, batch, self._embedding_dim)
        state_tuple = self.init_hidden(batch)
        output, state = self._encoder(obs_traj_embedding, state_tuple)
        final_h = state[0]

        return final_h


class AgentDecoderLSTM(nn.Module):
    '''
    This part of the code is revised from Social GAN's paper for fair comparison
    '''
    def __init__(self, seq_len, output_dim = 2, embedding_dim=32, h_dim=32, num_layers=1, dropout=0.0):
        super(AgentDecoderLSTM, self).__init__()

        self._seq_len = seq_len
        self._h_dim = h_dim
        self._embedding_dim = embedding_dim

        self._decoder = nn.LSTM(embedding_dim, h_dim, num_layers, dropout=dropout)
        self._spatial_embedding = nn.Linear(output_dim, embedding_dim)
        self._hidden2pos = nn.Linear(h_dim, output_dim)

    def relative_to_abs(self, rel_traj, start_pos=None):
        """
        Inputs:
        - rel_traj: pytorch tensor of shape (seq_len, batch, 2)
        - start_pos: pytorch tensor of shape (batch, 2)
        Outputs:
        - abs_traj: pytorch tensor of shape (seq_len, batch, 2)
        """
        # in our case, start pos is always 0
        if start_pos is None:
            start_pos = torch.tensor(np.zeros((rel_traj.shape[1], rel_traj.shape[2]))).type(torch.cuda.FloatTensor)

        rel_traj = rel_traj.permute(1, 0, 2)
        displacement = torch.cumsum(rel_traj, dim=1)
        start_pos = torch.unsqueeze(start_pos, dim=1)
        abs_traj = displacement + start_pos.type(torch.cuda.FloatTensor)

        return abs_traj.permute(1, 0, 2)

    def forward(self, last_pos_rel, state_tuple, start_pos=None, start_vel=None):
        """
        Inputs:
        - last_pos_rel: Tensor of shape (batch, 2)
        - state_tuple: (hh, ch) each tensor of shape (num_layers, batch, h_dim)
        Output:
        - pred_traj: tensor of shape (self.seq_len, batch, 2)
        """
        batch = last_pos_rel.size(0)
        pred_traj_fake_rel = []
        decoder_input = self._spatial_embedding(last_pos_rel)
        decoder_input = decoder_input.view(1, batch, self._embedding_dim)

        for _ in range(self._seq_len):
            output, state_tuple = self._decoder(decoder_input, state_tuple)
            rel_pos = self._hidden2pos(output.view(-1, self._h_dim))
            embedding_input = rel_pos
            decoder_input = self._spatial_embedding(embedding_input)
            decoder_input = decoder_input.view(1, batch, self._embedding_dim)
            pred_traj_fake_rel.append(rel_pos.view(batch, -1))

        pred_traj_fake_rel = torch.stack(pred_traj_fake_rel, dim=0)
        
        return self.relative_to_abs(pred_traj_fake_rel, start_pos), state_tuple[0]
 

class Classifier(nn.Module):
    '''
    Tianyang: GAN D Classifier
    '''
    def __init__(self, embed_dim_agent, classifier_hidden, dropout):
        super(Classifier, self).__init__()
        self._classifier = nn.Sequential(
                nn.Linear(embed_dim_agent, classifier_hidden),
                nn.LeakyReLU(),
                nn.Dropout(p=dropout),
                nn.Linear(classifier_hidden, 1),
                nn.Sigmoid()
                )

    def forward(self, x):
        return self._classifier(x)


# get noise, This part of the code is revised from Social GAN's paper for fair comparison
def get_noise(shape, noise_type='gaussian'):
    if noise_type == 'gaussian':
        return torch.randn(*shape).cuda()
    elif noise_type == 'uniform':
        return torch.rand(*shape).sub_(0.5).mul_(2.0).cuda()
    raise ValueError('Unrecognized noise type "%s"' % noise_type)

# make mlp, This part of the code is revised from Social GAN's paper for fair comparison
def make_mlp(dim_list, activation='leakyrelu', batch_norm=False, dropout=0):
    layers = []
    for dim_in, dim_out in zip(dim_list[:-1], dim_list[1:]):
        layers.append(nn.Linear(dim_in, dim_out))
        if batch_norm:
            layers.append(nn.BatchNorm1d(dim_out))
        if activation == 'relu':
            layers.append(nn.ReLU())
        elif activation == 'leakyrelu':
            layers.append(nn.LeakyReLU())
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
    return nn.Sequential(*layers)



###########################################################################################
##                                                                                       ##
##                                         Model                                         ##
##                                                                                       ##
###########################################################################################

class MultiAgentScene(nn.Module):
    '''
    Tianyang:
    MATF model
    '''
    def __init__(self, image_channels = 3, agent_indim = 2, \
            agent_outdim = 2, npast = 8, nfuture = 12, embed_dim_image = 32, 
            embed_dim_agent = 32, embed_image_h = 30, embed_image_w = 30, 
            spatial_embedding_linear_hidden_dim = 512, LSTM_layers = 1, dropout = 0.3, 
            classifier_hidden = 512, noise_dim = 16):

        super(MultiAgentScene, self).__init__()

        LSTM_layers = 1
        print('LSTM layer set to 1.')

        self._embed_dim_agent = embed_dim_agent
        self._embed_dim_image = embed_dim_image
        self._embed_image_h = embed_image_h
        self._embed_image_w = embed_image_w
        self._noise_dim = noise_dim
        self._agent_indim = agent_indim
        self._LSTM_layers = LSTM_layers

        self._semantic_image_encoder = SemanticImageEncoder(in_channels = image_channels, out_channels = embed_dim_image)
        self._spatial_pool_agent = SpatialPoolAgent()
        self._spatial_fetch_agent = SpatialFetchAgent(encoding_dim = embed_dim_agent)
        self._agent_map_fusion = AgentsMapFusion(in_channels = embed_dim_image + embed_dim_agent, out_channels = embed_dim_agent)
        self._agent_encoder_lstm = AgentEncoderLSTM(input_dim = agent_indim, embedding_dim = embed_dim_agent, h_dim = embed_dim_agent, \
            mlp_dim = spatial_embedding_linear_hidden_dim, num_layers = LSTM_layers, dropout = dropout)
        self._agent_decoder_lstm = AgentDecoderLSTM(seq_len = nfuture, output_dim = agent_outdim, embedding_dim = embed_dim_agent + noise_dim, \
            h_dim = embed_dim_agent + noise_dim, num_layers = LSTM_layers, dropout = dropout)
        self._classifier = Classifier(embed_dim_agent = embed_dim_agent, classifier_hidden = classifier_hidden, dropout = dropout)
        self._resnet = resnetShallow()

        print('Multi agent scene model initiated.')


    def list2batch(self, seq):
        # assemble a list of elements to batch, batch_idx: 0th dimension
        stacked = torch.tensor(seq[0]).unsqueeze(0)
        i = 1
        l = len(seq)
        while i < l:
            stacked = torch.cat((stacked, torch.tensor(seq[i]).unsqueeze(0)), 0)
            i += 1
        return stacked

    def batch2list(self, batch):
        # dis-assemble batch (index 0) to a list of elements
        unstacked = torch.unbind(batch, 0)
        return unstacked


    def load_from_pretrained_deterministic(self, path = 'outputs/stanford/state_dict.pt'):
        print('Multi Agent Scene Warning: You are trying to load state dicts. Please make sure that'
            + ' current file corresponds the file you intend to load:  ', path)

        pretrained_dict = torch.load(path)
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

        difference = []
        for k in model_dict:
            if k not in pretrained_dict:
                difference.append(k)
        print(difference)
        print('Reload difference shown in G.')

        model_dict.update(pretrained_dict) 
        self.load_state_dict(model_dict)
        print('Generator parameters loaded from pretrained model.')


    def forward(self, config, num_scenes, input_list, resample=0, std=1, use_resnet=0):
    	# config should be the architecture intended to run,
    	# which should be in ['baseline', 'single_agent_scene', 'multi_agent', 'multi_agent_scene', 'GAN_D', 'GAN_G', 'jointGAN_D', 'jointGAN_G']
        # input_list should be a list, whose each element comes from a scene:
        # each element is a list of [scene_id, agent_id, scene_image, number_agents, past_list, future_list, 
                                        # weights_list, coordinates_list, lanes, absolute_coordinate_list]

        ## format input
        num_agents_list = []
        input_image_list = []
        past_agents_state_list  = []
        future_agents_state_list  = []
        input_coordinate_list = []

        for i in range(num_scenes):
            num_agents_list.append(input_list[i][3])
            input_image_list.append(input_list[i][2][0])
            past_agents_state_list.append(input_list[i][4])
            future_agents_state_list.append(input_list[i][5])
            input_coordinate_list.append(input_list[i][7])

        num_agents = sum(num_agents_list)


        ## encode agents
        all_agents_past_list = []
        for i in range(num_scenes):
            for j in range(num_agents_list[i]):
                all_agents_past_list.append(past_agents_state_list[i][j])
        all_agents_past_batch = self.list2batch(all_agents_past_list)

        all_agents_future_list = []
        for i in range(num_scenes):
            for j in range(num_agents_list[i]):
                all_agents_future_list.append(future_agents_state_list[i][j])
        all_agents_future_batch = self.list2batch(all_agents_future_list) 

        if config in ['jointGAN_D', 'GAN_D']:
            all_agents_batch = torch.cat((all_agents_past_batch, all_agents_future_batch), 2)
        else:
            all_agents_batch = all_agents_past_batch

        all_agents_batch = all_agents_batch.permute(2, 0, 1)
            # permute for LSTM input order

        encoder_final_h_batch = self._agent_encoder_lstm(all_agents_batch.cuda())
        agents_indiv_batch = encoder_final_h_batch.view(num_agents, self._embed_dim_agent)
        agents_indiv_list = self.batch2list(agents_indiv_batch)


        ## encode image
        
        if config in ['single_agent_scene', 'multi_agent_scene', 'GAN_G', 'jointGAN_D', 'jointGAN_G']:
            input_image_batch = self.list2batch(input_image_list)
            if use_resnet == 0:
                embed_image_batch = self._semantic_image_encoder(input_image_batch.cuda())
            else:
                embed_image_batch = self._resnet(input_image_batch.cuda())
        
        elif config in ['multi_agent']:
            embed_image_batch = torch.tensor(np.zeros((num_scenes, self._embed_dim_image, \
                                        self._embed_image_h, self._embed_image_w), np.float32))

        
        ### spatial inference module
        if config in ['single_agent_scene', 'multi_agent', 'multi_agent_scene', 'GAN_G', 'jointGAN_D', 'jointGAN_G']:

            ## place and pool agents into a spatial map, which is inited as 0; iter on agents
            pooled_agents_map_batch = torch.tensor(np.zeros((num_scenes, self._embed_dim_agent, \
                                                self._embed_image_h, self._embed_image_w), np.float32))
            scene_idx = 0
            agent_in_scene_idx = 0

            for agent_indiv in agents_indiv_list:
                pooled_agents_map_batch = self._spatial_pool_agent(input_grid = pooled_agents_map_batch, \
                    input_state = agent_indiv.view(1, self._embed_dim_agent, 1), \
                    coordinate = input_coordinate_list[scene_idx][agent_in_scene_idx], batch_idx = scene_idx)

                agent_in_scene_idx += 1
                if agent_in_scene_idx >= num_agents_list[scene_idx]:
                    # move on to next scene
                    scene_idx += 1
                    agent_in_scene_idx = 0


            ## concat pooled agents map and embed image, reason on the joint grid
            fused_grid_batch = self._agent_map_fusion(input_agent = pooled_agents_map_batch.cuda(), \
                                                        input_map = embed_image_batch.cuda())


            ## fetch fused agents states back w.r.t. coordinates from fused map
            agents_fused_list = []
            agent_idx = 0

            for i in range(num_scenes):
                for j in range(num_agents_list[i]):
                    individual_state = agents_indiv_list[agent_idx].view(1, self._embed_dim_agent, 1)
                    agent_fused = self._spatial_fetch_agent(fused_scene = fused_grid_batch, individual_state = individual_state, \
                        coordinate = input_coordinate_list[i][j], batch_idx = i, pretrain = False)
                    agent_idx += 1
                    agents_fused_list.append(agent_fused.view(self._embed_dim_agent))


        ## final agent encodings, shape, a list of (self._embed_dim_agent,)
        if config in ['single_agent_scene', 'multi_agent', 'multi_agent_scene', 'GAN_G', 'jointGAN_D', 'jointGAN_G']:
            final_agents_encoding_list = agents_fused_list
        else:
            final_agents_encoding_list = agents_indiv_list


        ## classification for D of GANs, a list of scores
        if config in ['GAN_D', 'jointGAN_D']:
            final_agents_encoding_batch = self.list2batch(final_agents_encoding_list)
            classified = self._classifier(final_agents_encoding_batch.cuda())
            return classified, all_agents_future_batch, num_agents


        ## prediction of future trajectories, using decoder
        else:

            if resample == 0:

                # concat with noise
                noise = get_noise(shape=(self._noise_dim,), noise_type='gaussian')
                if config not in ['GAN_G', 'jointGAN_G']:
                    noise = 0.0 * noise

                all_agents_last_rel = (all_agents_past_batch[:, :, -1] \
                                                 - all_agents_past_batch[:, :, -2])\
                                                .view(num_agents, self._agent_indim)
                        # relative position of the last time stamp in past

                noised_list = []
                for agent in final_agents_encoding_list:
                    noised_agent = torch.cat((agent, noise), 0)
                    noised_list.append(noised_agent)
                decoder_h = self.list2batch(noised_list).view(1, num_agents, self._embed_dim_agent + self._noise_dim).cuda()

                decoder_c = torch.zeros(1, num_agents, self._embed_dim_agent + self._noise_dim).cuda()
                state_tuple = (decoder_h, decoder_c)

                # decode
                decoded, final_decoder_h = self._agent_decoder_lstm(last_pos_rel = all_agents_last_rel.cuda(), \
                                                                    state_tuple = state_tuple, \
                                                                    start_pos = all_agents_past_batch[:, :, -1], \
                                                                    start_vel = all_agents_past_batch[:, :, -1] - \
                                                                                all_agents_past_batch[:, :, -2])
                decoded = decoded.permute(1, 2, 0)
                return decoded, all_agents_future_batch, num_agents


            # resample for validation evaluation for GANs
            else:

                outputs_samples = []
                for resample_it in range(resample):

                    # concat with noise
                    noise = std * get_noise(shape=(self._noise_dim,), noise_type='gaussian')

                    all_agents_last_rel = (all_agents_past_batch[:, :, -1] \
                                                 - all_agents_past_batch[:, :, -2])\
                                                .view(num_agents, self._agent_indim)
                        # relative position of the last time stamp in past

                    noised_list = []
                    for agent in final_agents_encoding_list:
                        noised_agent = torch.cat((agent, noise), 0)
                        noised_list.append(noised_agent)
                    decoder_h = self.list2batch(noised_list).view(1, num_agents, self._embed_dim_agent + self._noise_dim).cuda()

                    decoder_c = torch.zeros(1, num_agents, self._embed_dim_agent + self._noise_dim).cuda()
                    state_tuple = (decoder_h.detach(), decoder_c.detach())
                        # no BP

                    # decode
                    decoded, final_decoder_h = self._agent_decoder_lstm(last_pos_rel = all_agents_last_rel.cuda(), \
                                                                        state_tuple = state_tuple, \
                                                                        start_pos = all_agents_past_batch[:, :, -1], \
                                                                        start_vel = all_agents_past_batch[:, :, -1] - \
                                                                                    all_agents_past_batch[:, :, -2])
                    decoded = decoded.permute(1, 2, 0)

                    outputs_samples.append(decoded.detach())
                return outputs_samples, all_agents_future_batch, num_agents
