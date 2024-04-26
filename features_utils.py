import torch
import numpy as np
from numpy.linalg import norm

def to_polar(point):
    if point[1] == 0 and point[0] == 0:
        return np.array([0.0, 0.0])
    else:    
        return np.array([np.sqrt(point[0]**2 + point[1]**2), np.arctan2(point[1], point[0])])

def parallel_polar(x):
    return [to_polar(i) for i in x]

def get_proj_point(u, x):
    # https://textbooks.math.gatech.edu/ila/projections.html
    proj_point = (u.dot(x) / u.dot(u)) * u
    return proj_point

def get_angle(v1, v2):
    if norm(v1) == 0 or norm(v2) == 0:
        return np.array(0.0)
    else:
        return np.arccos(np.clip(v1.dot(v2) / (norm(v1) * norm(v2)), -1.0, 1.0))

def get_risk_feature(ball_holder_pos, ball_pos, opp_pos, opp_body, candidate_point):
    pass_vector = candidate_point - ball_pos
    ball_to_opp = opp_pos - ball_pos

    opp_projection_point = get_proj_point(ball_to_opp, pass_vector) + ball_pos
    angle_ball_to_op_pass_vector = get_angle(ball_to_opp, pass_vector)
    
    proj_opp = opp_projection_point - opp_pos
    x_to_opp_proj = get_angle(proj_opp, np.array([1.0, 0.0]))
    new_opp_body = opp_body + x_to_opp_proj

    return [
        (angle_ball_to_op_pass_vector - new_opp_body, norm(ball_holder_pos - opp_pos)),
        norm(ball_to_opp), 
        norm(opp_projection_point - opp_pos),
        angle_ball_to_op_pass_vector,
        new_opp_body,
        norm(proj_opp - ball_holder_pos),
    ]

def parallel_get_risk(zipped):
    ballholder_pos, ball_pos, opp_cartesian_pos, body_angle, candidate_point = zipped
    return [
        ( 
            i, 
            get_risk_feature(ballholder_pos, ball_pos, opp_cartesian_pos[i], body_angle[i], candidate_point)
        ) for i in range(11)
    ]

def vector_list_distance(u, v):
    c = u.unsqueeze(3).repeat(1, 1, 1, 11).permute(0,1,3,2)
    ct = v.unsqueeze(3).repeat(1, 1, 1, 11).transpose(1,3).permute(0,1,3,2)
    diff = torch.sqrt(((c - ct)**2).sum(dim=(3)))

    return diff

def distance_to_line(line, p):
    norm = torch.sqrt((line**2).sum(dim=2, keepdim=True))
    proj = (((p*line).sum(dim=2, keepdim=True))*line)/(norm**2)

    perpendicular = proj - p

    return (perpendicular**2).sum(dim=2, keepdim=True)

def distance_to_line_segment(zero, target, p):
    v1 = target - zero
    v2 = p - target

    proj = (v1*v2).sum(dim=2, keepdim=True)
    norm = torch.sqrt((v1**2).sum(dim=len(v1.shape)-1, keepdim=True))

    return proj/norm