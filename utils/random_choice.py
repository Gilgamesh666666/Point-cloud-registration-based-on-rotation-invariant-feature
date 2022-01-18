import numpy as np
def randchoice(n, m):
    if n < m:
        idx = np.random.choice(n, m, replace = True)
    else:
        idx = np.random.choice(n, m, replace = False)
    return idx

def farthest_point_sample(pc, m):
    '''
    input: np float(n, 3)
    output: np float(m, 3)
    '''
    n = pc.shape[0]
    assert m < n
    dim = pc.shape[1]
    selected_pts = np.zeros((m, dim))
    selected_idx = np.zeros(m)
    selected_mask = np.ones(n, dtype=bool)
    selected_num = 0

    start_idx = np.randint(n, (1, )).long()
    selected_pts[0] = pc[start_idx]
    selected_idx[0] = start_idx
    selected_mask[start_idx] = 0
    
    new_point = selected_pts[selected_num, :]
    min_dist = np.ones(n, 1)*10000
    
    selected_num = selected_num + 1
    while(selected_num < m):
        # 求pc到sample set的距离(pc中的点到sample set中点的最小距离)
        selected_remain = pc[selected_mask]
        selected_remain_idx = np.arange(n)[selected_mask].long()
        now_dist = np.linalg.norm(selected_remain - new_point.unsqueeze(0), axis=1, keepdim=True)
        min_dist[selected_mask] = np.min(np.concatenate((now_dist, min_dist[selected_mask]), axis=1), axis=1, keepdim=True).values # (n,1)
        #print(pc2selected_dist.shape)
        #print((selected_remain.unsqueeze(1) - selected_pts[:selected_num].unsqueeze(0)).shape)
        # pc中到sample set的距离最大的点
        now_idx = np.argmax(min_dist[selected_mask],axis=0)  #(1, )
        selected_pts[selected_num] = selected_remain[now_idx]
        selected_idx[selected_num] = selected_remain_idx[now_idx]
        selected_mask[now_idx] = 0
        new_point = selected_pts[selected_num]
        selected_num = selected_num + 1
    return selected_pts, selected_idx