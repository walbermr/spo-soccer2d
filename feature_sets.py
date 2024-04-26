import torch
import numpy as np
import multiprocessing

from features_utils import *


def get_featureset(constants, wm, action_row, normalize=False, candidate_point=None, fset_fn=None):
    action_row = np.delete(action_row, (0, 1), axis=1) #removes hash and action class
    wm = np.delete(wm, (0), axis=1)
    
    if candidate_point is None:
        candidate_point = torch.tensor(np.array([action_row[:, 3], action_row[:, 4]]).T).unsqueeze(1)
    
    new_frame = fset_fn(constants, wm, candidate_point, normalize)
    dropped_action_df = np.delete(action_row, (0, 1, 2, 5, 6, 7, 8), axis=1) #removes hash and action class.

    return new_frame, dropped_action_df


def get_tensor_idx(multiplier, offset, iterator, shape, device):
    return torch.tensor([[i*multiplier+offset, i*multiplier+offset+1] for i in iterator]).to(device).reshape(-1).unsqueeze(0).repeat(shape, 1)


def fset_1(constants, wm, candidate_point, normalize=False, normalize_candidate_point=True):
    player_features = wm[:, constants._i_delta:]

    if isinstance(wm, torch.Tensor):
        device = wm.device
        bs = wm.shape[0]
        ball_pos = torch.stack([wm[:, 2], wm[:, 3]]).T.unsqueeze(1)
        gather_idx = get_tensor_idx(6, 1, range(11), bs, device)
        tm_cartesian_pos = torch.gather(player_features, 1, gather_idx).unsqueeze(1).reshape(-1,11,2)
        
        gather_idx = get_tensor_idx(6, 1, range(11,22), bs, device)
        opp_cartesian_pos = torch.gather(player_features, 1, gather_idx).unsqueeze(1).reshape(-1,11,2)
    else:
        ball_pos = torch.stack([torch.tensor(wm[:, 2]), torch.tensor(wm[:, 3])]).T.unsqueeze(1)
        #TODO verificar se era pra fazer o cast para torch stack, aq, acho que ta errado em cima.
        tm_cartesian_pos = torch.tensor(
            np.stack(
                    [np.stack([player_features[:, (i*6)+1], player_features[:,(i*6+2)]]) for i in range(11)]
                )
        ).permute(2,0,1) #x,y for each player
        
        opp_cartesian_pos = torch.tensor(
            np.stack(
                    [np.stack([player_features[:, (i*6)+1], player_features[:,(i*6+2)]]) for i in range(11,22)]
                )
        ).permute(2,0,1) #x,y for each player

    device = ball_pos.device
    
    if normalize:
        pos_norm_factor = torch.tensor([[constants._pitch_width_norm/2], [constants._pitch_height_norm/2]], device=device).transpose(1,0)
        vel_norm_factor = torch.tensor([constants._player_max_vel_norm], device=device)
        ball_pos /= pos_norm_factor
        tm_cartesian_pos /= pos_norm_factor
        opp_cartesian_pos /= pos_norm_factor

        if normalize_candidate_point:
            candidate_point /= pos_norm_factor
    
    pass_vector = candidate_point - ball_pos

    player_distances = vector_list_distance(tm_cartesian_pos, opp_cartesian_pos)
    opp_distance_to_pass = distance_to_line(pass_vector.repeat(1,11,1), opp_cartesian_pos).permute(0,2,1)
    tm_distance_to_pass = distance_to_line(pass_vector.repeat(1,11,1), tm_cartesian_pos).permute(0,2,1)
    tm_distance_to_target = (candidate_point - tm_cartesian_pos).sum(dim=2, keepdim=True).permute(0,2,1)
    opp_distance_to_target = (candidate_point - opp_cartesian_pos).sum(dim=2, keepdim=True).permute(0,2,1)

    new_frame = torch.cat(
        [
            player_distances,
            opp_distance_to_pass, 
            tm_distance_to_pass, 
            tm_distance_to_target, 
            opp_distance_to_target,
        ]
        , dim=1
    ).flatten(start_dim=1)
    
    new_frame = torch.cat([new_frame, candidate_point.flatten(start_dim=1)], dim=1)

    return new_frame


def fset_2(constants, wm, candidate_point, normalize=False, normalize_candidate_point=True):
    player_features = wm[:, constants._i_delta:]

    if isinstance(wm, torch.Tensor):
        device = wm.device
        bs = wm.shape[0]
        ball_pos = torch.stack([wm[:, 2], wm[:, 3]]).T.unsqueeze(1)
        gather_idx = get_tensor_idx(6, 1, range(11), bs, device)
        tm_cartesian_pos = torch.gather(player_features, 1, gather_idx).unsqueeze(1).view(-1,11,2)
        
        gather_idx = get_tensor_idx(6, 1, range(11,22), bs, device)
        opp_cartesian_pos = torch.gather(player_features, 1, gather_idx).unsqueeze(1).view(-1,11,2)
    else:
        ball_pos = torch.stack([torch.tensor(wm[:, 2]), torch.tensor(wm[:, 3])]).T.unsqueeze(1)
        #tm_cartesian_pos = torch.gather(player_features, 1, torch.tensor([[i*6+1, i*6+2] for i in range(11)]))
        tm_cartesian_pos = torch.tensor(
            np.array(
                [np.array([player_features[:, (i*6)+1], player_features[:, (i*6)+2]]) for i in range(11)]
            )
        ).permute(2,0,1) #x,y for each player
        #opp_cartesian_pos = torch.gather(player_features, 1, torch.tensor([[i*6+1, i*6+2] for i in range(11,22)]))
        opp_cartesian_pos = torch.tensor(
            np.array(
                [np.array([player_features[:, (i*6)+1], player_features[:, (i*6)+2]]) for i in range(11,22)]
            )
        ).permute(2,0,1) #x,y for each player
    
    goal_pos1 = constants.goal_pos1
    goal_pos2 = constants.goal_pos2
    
    if normalize:
        pos_norm_factor = constants.pos_norm_factor
        vel_norm_factor = constants.vel_norm_factor
        ball_pos /= pos_norm_factor
        tm_cartesian_pos /= pos_norm_factor
        opp_cartesian_pos /= pos_norm_factor
        goal_pos1 = constants.norm_goal_pos1.flatten()
        goal_pos2 = constants.norm_goal_pos2.flatten()
        
        if normalize_candidate_point:
            candidate_point /= pos_norm_factor

    player_distances = vector_list_distance(tm_cartesian_pos, opp_cartesian_pos)
    opp_distance_to_pass = distance_to_line_segment(
        ball_pos.repeat(1,11,1), 
        candidate_point.expand(-1,11,-1), 
        opp_cartesian_pos
    ).permute(0,2,1)
    tm_distance_to_pass = distance_to_line_segment(
        ball_pos.repeat(1,11,1), 
        candidate_point.expand(-1,11,-1), 
        tm_cartesian_pos
    ).permute(0,2,1)
    tm_distance_to_goal_line = torch.exp(
        -distance_to_line_segment(
            goal_pos1,
            goal_pos2,
            tm_cartesian_pos,
        ).permute(0,2,1)
    )
    tm_distance_to_target = (candidate_point - tm_cartesian_pos).sum(dim=2, keepdim=True).permute(0,2,1)
    opp_distance_to_target = (candidate_point - opp_cartesian_pos).sum(dim=2, keepdim=True).permute(0,2,1)

    new_frame = torch.cat([
            player_distances,
            opp_distance_to_pass, 
            tm_distance_to_pass, 
            tm_distance_to_target, 
            opp_distance_to_target,
            tm_distance_to_goal_line
        ], dim=1
    ).flatten(start_dim=1)
    new_frame = torch.cat([new_frame, candidate_point.flatten(start_dim=1)], dim=1)

    return new_frame


def positional_fset(constants, wm, candidate_point, normalize=False, normalize_candidate_point=True):
    player_features = wm[:, constants._i_delta:]

    if isinstance(wm, torch.Tensor):
        device = wm.device
        bs = wm.shape[0]
        ball_pos = torch.stack([wm[:, 2], wm[:, 3]]).T.unsqueeze(1)
        gather_idx = get_tensor_idx(6, 1, range(11), bs, device)
        tm_cartesian_pos = torch.gather(player_features, 1, gather_idx).unsqueeze(1).view(-1,11,2)
        
        gather_idx = get_tensor_idx(6, 1, range(11,22), bs, device)
        opp_cartesian_pos = torch.gather(player_features, 1, gather_idx).unsqueeze(1).view(-1,11,2)

        gather_idx = get_tensor_idx(6, 3, range(11), bs, device)
        tm_cartesian_vel = torch.gather(player_features, 1, gather_idx).unsqueeze(1).view(-1,11,2)
        
        gather_idx = get_tensor_idx(6, 3, range(11,22), bs, device)
        opp_cartesian_vel = torch.gather(player_features, 1, gather_idx).unsqueeze(1).view(-1,11,2)
        # body_angle = np.stack([player_features[:, (i*6)+5] for i in range(22)]).transpose(1,0)
    else:
        ball_pos = torch.tensor(np.stack([wm[:, 2], wm[:, 3]]).T).unsqueeze(1)
        tm_cartesian_pos = np.stack([[player_features[:, (i*6)+1], player_features[:, (i*6)+2]] for i in range(11)]).transpose(2,0,1) #x,y for each player
        opp_cartesian_pos = np.stack([[player_features[:, (i*6)+1], player_features[:, (i*6)+2]] for i in range(11,22)]).transpose(2,0,1) #x,y for each player
        tm_cartesian_vel = np.stack([[player_features[:, (i*6)+3], player_features[:, (i*6)+4]] for i in range(11)]).transpose(2,0,1) #x,y for each player
        opp_cartesian_vel = np.stack([[player_features[:, (i*6)+3], player_features[:, (i*6)+4]] for i in range(11,22)]).transpose(2,0,1) #x,y for each player
        # body_angle = np.stack([player_features[:, (i*6)+5] for i in range(22)]).transpose(1,0)
    
    device = ball_pos.device

    if normalize:
        pos_norm_factor = torch.tensor([[constants._pitch_width_norm/2], [constants._pitch_height_norm/2]], device=device).transpose(1,0)
        vel_norm_factor = torch.tensor([constants._player_max_vel_norm], device=device)
        ball_pos /= pos_norm_factor
        tm_cartesian_pos /= pos_norm_factor
        opp_cartesian_pos /= pos_norm_factor
        tm_cartesian_vel /= vel_norm_factor
        opp_cartesian_vel /= vel_norm_factor

        if normalize_candidate_point:
            candidate_point /= pos_norm_factor
    
    new_frame = torch.cat(
        [
            ball_pos,
            candidate_point, 
            tm_cartesian_pos,
            opp_cartesian_pos,
        ]
        , dim=1
    ).flatten(start_dim=1)

    return new_frame