import numpy as np
import json
import os
import random
import math
import time
import sys
import copy
from joblib import Parallel, delayed
from collections import deque
from itertools import combinations
from collections import Counter
import glob


os.environ['OMP_NUM_THREADS'] = '1'

from sklearn.cluster import KMeans
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.visualization.scatter import Scatter # 仅用于最后的可视化

import torch
from dqn_utils import DQNAgent, device
# ================= 核心修改: 导入新的高级局部搜索算子 =================
import advanced_local_search as als

USE_DQN = True

# =================================================================================
# 问题数据加载
# =================================================================================
def load_problem_data(file_path):
    """
    从JSON文件加载并解析问题数据 - 与Gurobi求解器完全一致
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    q = {int(k): v for k, v in data['task_demand'].items()}
    tau = {int(k): v for k, v in data['task_service_time'].items()}
    coords = {int(k): tuple(v) for k, v in data['coordinates'].items()}

    dist = {}
    for k, v in data['distance_matrix_str_keys'].items():
        i, j = map(int, k.split(','))
        dist[i, j] = v

    K = data['num_robots']
    M = data['num_tasks']
    N = data['num_nodes']
    Q = data['robot_capacity']
    v_base = data['robot_speed']
    P = [tuple(p) for p in data['task_precedence']]

    # BUGFIX: 确保距离矩阵包含对角线元素(i,i)，特别是(0,0)，以避免KeyError
    for i in range(0, M + 1):
        dist.setdefault((i, i), 0.0)

    # ========================== 健壮性BUG修复 ==========================
    # 确保 Q 和 v_base 总是列表格式，以统一处理
    if not isinstance(Q, list):
        Q = [Q] * K
    if not isinstance(v_base, list):
        v_base = [v_base] * K
    # =================================================================

    # 像Gurobi一样，为问题补充默认时间窗，以统一处理等待时间
    TW = {i: (0, 1000) for i in range(1, M + 1)}

    # 与Gurobi完全一致的参数设置
    e_t = {i: 5.0 for i in range(1, M + 1)}
    e_s = 10.0
    e_f = 0.03
    ef = {r: 1.0 for r in range(1, K + 1)}
    base_energy_per_dist_per_speed = 0.005

    return {
        'K': K, 'M': M, 'N': N, 'Q': Q, 'v_base': v_base, 'q': q, 'tau': tau,
        'P': P, 'coords': coords, 'dist': dist,
        'e_t': e_t, 'e_s': e_s, 'e_f': e_f, 'ef': ef,
        'base_energy_per_dist_per_speed': base_energy_per_dist_per_speed,
        'TW': TW
    }


def get_adaptive_decoder_params(params):
    """根据算例特性自动计算解码器参数，无需手动调优"""
    M, K = params['M'], params['K']
    
    # TOP_K基于问题规模自适应：大问题需要限制候选数以提高速度
    if M <= 20:
        TOP_K = 4
    elif M <= 50:
        TOP_K = 3
    else:  # M > 50，大规模问题
        TOP_K = 2  # 进一步减少候选数以提高速度
    
    # 温度基于问题规模自适应：大问题用更高温度补偿TOP_K的减少
    if M <= 30:
        TEMPERATURE = 0.5
    elif M <= 70:
        TEMPERATURE = 0.7
    else:  # M > 70
        TEMPERATURE = 0.9  # 大问题用极高随机性以增加多样性
    
    # 噪声保持适中
    EPS_NOISE = 1e-4
    
    return TOP_K, TEMPERATURE, EPS_NOISE

def _softmax_choice(candidates, temperature):
    """用负cost做softmax，cost越小概率越大"""
    import math, random
    costs = np.array([c[0] for c in candidates], dtype=float)
    logits = -costs / max(1e-9, temperature)
    exps = np.exp(logits - logits.max())
    probs = exps / exps.sum()
    return candidates[np.random.choice(len(candidates), p=probs)][1]

def _add_simple_diversity(cost, individual_id, task, r_id, pos):
    """增强的多样性机制：为不同个体引入更强的差异"""
    # 基于个体ID的哈希扰动，增加扰动强度
    hash_val = hash((individual_id, task, r_id, pos)) & 0xffffffff
    noise_factor = (hash_val % 1000) / 1000.0  # 0-1之间的伪随机数
    
    # 增加扰动强度，特别是对后期个体ID
    base_noise = 0.05  # 提高基础扰动到5%
    id_noise = 0.002 * (individual_id % 100)  # 根据个体ID增加更多额外扰动
    total_noise = base_noise + id_noise
    
    return cost * (1 + total_noise * (noise_factor - 0.5))  # 增强扰动
def decode_chromosome(permutation, params):
    
    # 获取自适应参数
    TOP_K_INSERT, SOFTMAX_TEMPERATURE, BASE_EPS_NOISE = get_adaptive_decoder_params(params)
    individual_id = params.get('current_individual_id', 0)
    
   
    current_gen = params.get('current_generation', 0)
    last_shown_gen = params.get('_last_decoder_print_gen', -1)
    if individual_id == 0 and current_gen != last_shown_gen:
        print(f"[Decoder Gen {current_gen}] TOP_K={TOP_K_INSERT}, TEMP={SOFTMAX_TEMPERATURE:.2f}")
        params['_last_decoder_print_gen'] = current_gen
    routes = {k: [] for k in range(1, params['K'] + 1)}
    
    # 预处理先后约束
    predecessors = {i: set() for i in range(1, params['M'] + 1)}
    for u, v in params['P']:
        predecessors[v].add(u)

    # 模拟每个机器人的状态
    robot_states = {r: {'path': [], 'time': 0.0} for r in range(1, params['K'] + 1)}
    
    # 跟踪已完成任务的时间戳
    finish_times = {}
    
    tasks_to_assign = list(permutation)
    
    while tasks_to_assign:
        candidate_tasks = [
            t for t in tasks_to_assign
            if predecessors[t].issubset(finish_times.keys())
        ]
        
        if not candidate_tasks:
            print(f"[严重警告] 解码时出现约束死锁，无法找到下一个可执行的任务。剩余任务: {tasks_to_assign}")
            # 作为备用策略，将剩余任务随机分配，但这可能会产生一个不可行的解
            for task in tasks_to_assign:
                r = random.randint(1, params['K'])
                routes[r].append(task)
            break

        # GRASP化解码：寻找当前轮次的多个候选插入，然后软选择
        best_global_candidates = []  # [(cost, insertion_dict), ...]
        min_global_cost = float('inf')

        # 从优先级最高的候选任务开始评估
        for task in (p for p in permutation if p in candidate_tasks):
            # 收集该task在不同机器人/位置的可行插入
            local_candidates = []

            for r_id in range(1, params['K'] + 1):
                # 检查容量
                current_load = sum(params['q'][t] for t in routes[r_id])
                if current_load + params['q'][task] > params['Q'][r_id-1]:
                    continue

                # 检查所有可能的插入位置
                for i in range(len(routes[r_id]) + 1):
                    temp_route = routes[r_id][:i] + [task] + routes[r_id][i:]
                    
                    # 快速模拟时间，检查先后约束（优化版本）
                    is_feasible = True
                    s_task, f_task = 0, 0
                    t_cur = 0.0
                    
                    path_sim = [0] + temp_route + [0]
                    for j in range(len(path_sim) - 1):
                        a, b = path_sim[j], path_sim[j+1]
                        travel = params['dist'][(a, b)] / params['v_base'][r_id-1]
                        arrive = t_cur + travel
                        
                        if b != 0:
                            s_start = max(arrive, params['TW'][b][0])
                            
                            # 快速前置约束检查：一旦发现违反立即退出
                            if b in predecessors:
                                for pred_task in predecessors[b]:
                                    if pred_task in finish_times and s_start < finish_times[pred_task]:
                                        is_feasible = False
                                        break
                            if not is_feasible: 
                                break  # 立即退出整个路径模拟

                            f_end = s_start + params['tau'][b]
                            t_cur = f_end
                            if b == task:
                                s_task, f_task = s_start, f_end
                        else:
                            t_cur = arrive
                    
                    if is_feasible:
                        # 改进的成本计算：考虑负载均衡
                        original_finish_time = robot_states[r_id]['time']
                        time_cost = t_cur - original_finish_time
                        
                        # 计算当前机器人的任务负载
                        current_task_count = len(routes[r_id])
                        avg_tasks_per_robot = len(candidate_tasks) / params['K']
                        
                        # 负载均衡惩罚：任务数偏离平均值的惩罚
                        load_imbalance_penalty = abs(current_task_count + 1 - avg_tasks_per_robot) * 100
                        
                        # 时间负载惩罚：鼓励将时间密集型任务分配给快速机器人
                        time_load_penalty = params['tau'][task] / params['v_base'][r_id-1] * 0.5
                        
                        # 综合成本
                        cost = time_cost + load_imbalance_penalty + time_load_penalty
                        
                        # ===== 简化的多样性机制 =====
                        # 只保留最必要的个体特异性扰动
                        cost = _add_simple_diversity(cost, individual_id, task, r_id, i)
                        # =================================

                        local_candidates.append((cost, {
                            'task': task, 'robot': r_id, 'pos': i,
                            'new_finish_time': t_cur, 's_task': s_task, 'f_task': f_task
                        }))

            if local_candidates:
                # 取前K个局部最优
                local_candidates.sort(key=lambda x: x[0])
                local_topk = local_candidates[:TOP_K_INSERT]
                # 保留到全局候选池
                best_global_candidates.extend(local_topk)
                # 更新全局最小cost，仅用于统计（可选）
                min_global_cost = min(min_global_cost, local_topk[0][0])
                
                # 大规模问题的额外限制：如果全局候选池太大，立即处理
                max_candidates = 30 if params['M'] > 70 else 50
                if len(best_global_candidates) > max_candidates:
                    break

        # 执行本轮插入：不再取单一最优，而是对全局前K做一次软选择
        if best_global_candidates:
            best_global_candidates.sort(key=lambda x: x[0])
            global_topk = best_global_candidates[:TOP_K_INSERT]
            chosen = _softmax_choice(global_topk, SOFTMAX_TEMPERATURE)
            
            t = chosen['task']
            r = chosen['robot'] 
            pos = chosen['pos']
            
            routes[r].insert(pos, t)
            robot_states[r]['time'] = chosen['new_finish_time']
            finish_times[t] = chosen['f_task']
            tasks_to_assign.remove(t)
        else:
            # 如果所有候选任务在所有位置都不可行（极为罕见），则强制分配一个
            # 清除缓存的索引表，因为它只对当前排列有效
            if '___perm_to_idx' in locals():
                del ___perm_to_idx

            task_to_force = next(p for p in permutation if p in candidate_tasks)
            r = random.randint(1, params['K'])
            routes[r].append(task_to_force)
            tasks_to_assign.remove(task_to_force)
            # 注意: 此时未更新finish_times，可能会在后续评估中被惩罚，但避免了死循环
            print(f"[警告] 解码时找不到任何可行的插入点。强制分配任务 {task_to_force}。")

    # 清除为本次解码缓存的索引表
    if '___perm_to_idx' in locals():
        del ___perm_to_idx
        
    return routes


def calculate_single_route_properties(tasks, r_id, params):
    
    p = params
    if not tasks:
        return {'distance': 0, 'energy': 0, 'finish_time': 0}

    current_completion_time = 0
    robot_dist = 0
    robot_wait_time = 0

    path = [0] + tasks + [0]
    for i in range(len(path) - 1):
        start_node, end_node = path[i], path[i+1]
        segment_dist = p['dist'][(start_node, end_node)]
        robot_dist += segment_dist
        
        travel_time = segment_dist / p['v_base'][r_id-1]
        arrival_time = current_completion_time + travel_time
        
        service_start_time = arrival_time
        if end_node != 0:
            service_start_time = max(arrival_time, p['TW'][end_node][0])
        
        wait_time = service_start_time - arrival_time
        robot_wait_time += wait_time

        if end_node != 0:
            current_completion_time = service_start_time + p['tau'][end_node]
        else:
            current_completion_time = service_start_time

    driving_energy = (robot_dist / p['v_base'][r_id-1]) * p['base_energy_per_dist_per_speed'] * p['v_base'][r_id-1]**2 * p['ef'][r_id]
    task_energy = sum(p['e_t'][t] for t in tasks)
    wait_energy = robot_wait_time * p['e_s']
    fixed_energy = sum(p['e_f'] * p['tau'][t] for t in tasks)
    total_robot_energy = driving_energy + task_energy + wait_energy + fixed_energy
    
    return {
        'distance': robot_dist, 
        'energy': total_robot_energy, 
        'finish_time': current_completion_time
    }

def calculate_objectives_from_properties(route_properties):
    """
    从所有路径的属性字典中聚合计算最终的三个目标函数。
    """
    total_distance = sum(prop['distance'] for prop in route_properties.values())
    total_energy = sum(prop['energy'] for prop in route_properties.values())
    
    # 修改：与Gurobi完全一致，考虑所有机器人（包括未使用的，完成时间为0）
    all_finish_times = [prop['finish_time'] for prop in route_properties.values()]
    if all_finish_times:
        imbalance = max(all_finish_times) - min(all_finish_times)
    else:
        imbalance = 0
    
    return [total_distance, total_energy, imbalance]


def simulate_task_times(routes, params):
    
    start_time, finish_time = {}, {}
    p = params
    for r, tasks in routes.items():
        t_cur = 0.0
        path = [0] + tasks + [0]
        for i in range(len(path) - 1):
            a, b = path[i], path[i+1]
            dist = p['dist'][(a, b)]
            travel = dist / p['v_base'][r-1]
            arrive = t_cur + travel
            if b != 0:
                s_start = max(arrive, p['TW'][b][0])
                start_time[b] = s_start
                f = s_start + p['tau'][b]
                finish_time[b] = f
                t_cur = f
            else:
                t_cur = arrive
    return start_time, finish_time

def evaluate(routes, params, individual_id=None):
   
    # 核心改造：个体已经是 routes 字典，不再需要解码

    # 1. 检查所有任务是否都已被分配 (更严格的检查)
    # 将所有路径中的任务平铺到一个列表中，包括重复项
    flat_tasks = [task for path in routes.values() for task in path]
    # 检查: a) 独立任务的数量是否等于M  b) 任务总数（含重复）是否等于M
    # 这可以同时捕捉到任务丢失和任务重复分配两种错误
    if len(set(flat_tasks)) != params['M'] or len(flat_tasks) != params['M']:
        return [1e7, 1e7, 1e7]

    # 2. 检查容量约束
    for r, tasks in routes.items():
        if sum(params['q'][t] for t in tasks) > params['Q'][r-1]:
            return [1e7, 1e7, 1e7]

    # 3. 检查先后约束
    s, f = simulate_task_times(routes, params)
    precedence_violations = 0
    for u, v in params['P']:
        if u in f and v in s and f[u] > s[v]:
            precedence_violations += 1

    # 计算目标函数
    all_props = {r_id: calculate_single_route_properties(tasks, r_id, params) for r_id, tasks in routes.items()}
    base_objectives = calculate_objectives_from_properties(all_props)

    if precedence_violations > 0:
        PENALTY = 1e6 * precedence_violations
        return [
            base_objectives[0] + PENALTY,
            base_objectives[1] + PENALTY,
            base_objectives[2] + PENALTY
        ]
    
    return base_objectives

def evaluate_population(population, params, n_jobs=1, backend="threading"):
    
    if n_jobs == 1:
        # 串行模式 (推荐用于调试和Windows环境)
        return [evaluate(routes, params, individual_id=i) for i, routes in enumerate(population)]
    
    # 并行模式
    try:
        results = Parallel(n_jobs=n_jobs, backend=backend)(
            delayed(evaluate)(routes, params, individual_id=i)
            for i, routes in enumerate(population)
        )
        return results
    except Exception as e:
        print(f"[严重错误] 并行评估失败 (backend={backend}): {e}")
        print("切换到串行模式进行评估...")
        return [evaluate(routes, params, individual_id=i) for i, routes in enumerate(population)]



def calculate_crowding_distance(objectives, fronts):
    
    if not isinstance(objectives, np.ndarray) or objectives.ndim != 2:
        # 如果 objectives 为空或格式不正确，则返回一个空数组或根据情况处理
        if len(objectives) == 0:
            return np.array([])
        raise ValueError("目标必须是一个2D numpy 数组。")

    num_individuals = objectives.shape[0]
    crowding_distances = np.zeros(num_individuals)

    for front_indices in fronts:
        # Pymoo 返回的 front 是 numpy array
        if isinstance(front_indices, np.ndarray):
            front_indices = front_indices.tolist()

        if not front_indices or len(front_indices) == 0:
            continue

        front_objectives = objectives[front_indices, :]
        num_points = len(front_indices)
        num_obj = objectives.shape[1]

        # 创建一个用于存储当前前沿拥挤度的临时数组
        front_crowding = np.zeros(num_points)

        for m in range(num_obj):
            # 按当前目标排序
            sorted_indices_in_front = np.argsort(front_objectives[:, m])

            # 为边界点分配无限大的距离
            front_crowding[sorted_indices_in_front[0]] = np.inf
            front_crowding[sorted_indices_in_front[-1]] = np.inf

            # 获取目标值的最大和最小值
            f_max = front_objectives[sorted_indices_in_front[-1], m]
            f_min = front_objectives[sorted_indices_in_front[0], m]

            # 如果所有值都相同，则跳过
            if f_max == f_min:
                continue

            # 计算中间点的拥挤度
            for i in range(1, num_points - 1):
                original_index_in_front = sorted_indices_in_front[i]
                prev_obj_val = front_objectives[sorted_indices_in_front[i-1], m]
                next_obj_val = front_objectives[sorted_indices_in_front[i+1], m]
                
                front_crowding[original_index_in_front] += (next_obj_val - prev_obj_val) / (f_max - f_min)

        # 将当前前沿的拥挤度映射回主拥挤度数组
        for i in range(num_points):
            original_pop_index = front_indices[i]
            crowding_distances[original_pop_index] = front_crowding[i]
            
    return crowding_distances


def robust_non_dominated_sorting(objectives_array):
    
    unique_objectives_map = {}  # 映射: tuple(obj) -> [原始索引列表]
    for i, obj in enumerate(objectives_array):
        obj_tuple = tuple(obj)
        if obj_tuple not in unique_objectives_map:
            unique_objectives_map[obj_tuple] = []
        unique_objectives_map[obj_tuple].append(i)

    unique_objectives_list = np.array(list(unique_objectives_map.keys()))
    # unique_indices_list 是一个列表的列表，保存了每个独特解对应的所有原始个体索引
    unique_indices_list = list(unique_objectives_map.values())

    # 仅对独特的目标值执行NDS
    unique_fronts = NonDominatedSorting().do(unique_objectives_list)

    # 重建完整的前沿和排名映射
    reconstructed_fronts = []
    rank_map = {}
    for rank_idx, front in enumerate(unique_fronts):
        new_front = []
        for unique_idx_in_front in front:
            # 找到这个独特解对应的所有原始个体
            original_indices = unique_indices_list[unique_idx_in_front]
            new_front.extend(original_indices)
            for original_idx in original_indices:
                rank_map[original_idx] = rank_idx
        reconstructed_fronts.append(new_front)

    return reconstructed_fronts, rank_map


def _split_cluster_by_Q(cluster, Q_max, q):
   
    tasks = sorted(cluster, key=lambda t: q[t], reverse=True)
    bins, loads = [[]], [0.0]
    for t in tasks:
        placed = False
        for i in range(len(bins)):
            if loads[i] + q[t] <= Q_max:
                bins[i].append(t)
                loads[i] += q[t]
                placed = True
                break
        if not placed:
            bins.append([t])
            loads.append(q[t])
    return [b for b in bins if b] # 确保不返回空簇


def generate_heuristic_individual(params, seed):
   
    # ================== 阶段一: 任务分组 (K-Means++ Clustering) ==================
    tasks = list(range(1, params['M'] + 1))
    task_coords = np.array([params['coords'][t] for t in tasks])
    
    # 动态确定聚类数量 K
    total_demand = sum(params['q'].values())
    avg_capacity = np.mean(params['Q'])
    num_clusters = max(params['K'], math.ceil(total_demand / avg_capacity))
    num_clusters = min(num_clusters, params['M']) # 确保簇的数量不超过任务数量

    kmeans = KMeans(n_clusters=num_clusters, init='k-means++', n_init=10, random_state=None) # 解除随机种子以进行鲁棒性测试
    labels = kmeans.fit_predict(task_coords)

    clusters = [[] for _ in range(num_clusters)]
    for task_idx, label in enumerate(labels):
        clusters[label].append(tasks[task_idx])
    
    clusters = [c for c in clusters if c] # 移除可能出现的空簇

    # ================== 阶段二: 多级启发式任务组分配 ==================
    # 预计算每个簇的属性
    cluster_info = []
    for cluster in clusters:
        demand = sum(params['q'][t] for t in cluster)
        service_time = sum(params['tau'][t] for t in cluster)
        cluster_info.append({'tasks': cluster, 'demand': demand, 'service_time': service_time})

    # 按总需求量、总服务时间降序排序
    cluster_info.sort(key=lambda x: (x['demand'], x['service_time']), reverse=True)

    # 初始化机器人状态
    robot_states = [{'tasks': [], 'load': 0, 'last_pos': 0, 'time': 0} for _ in range(params['K'])]

    # 使用双端队列来处理，避免在迭代时修改列表
    clusters_to_process = deque(cluster_info)

    # 迭代分配任务组
    while clusters_to_process:
        cluster = clusters_to_process.popleft() # 从队列头部取出一个簇
        
        best_robot_idx = -1
        candidate_robots = []

        # 级别一: 可行性筛选
        feasible_robots = [
            i for i, r in enumerate(robot_states) 
            if r['load'] + cluster['demand'] <= params['Q'][i]
        ]
        
        if not feasible_robots:
            # 如果一个只包含单个任务的簇都无法分配，说明该任务需求超过任何单个机器人的容量，问题可能无解
            if len(cluster['tasks']) == 1:
                print(f"[警告] 任务 {cluster['tasks'][0]} (需求: {cluster['demand']}) 无法被任何机器人分配。初始化时将跳过此任务。")
                continue

            # 拆分过大的簇
            Q_max = max(params['Q'])
            sub_clusters_tasks = _split_cluster_by_Q(cluster['tasks'], Q_max, params['q'])
            
          
            if len(sub_clusters_tasks) == 1:
                continue

            # 将拆分后的新子簇加入队列头部优先处理
            for sub in reversed(sub_clusters_tasks):
                clusters_to_process.appendleft({
                    'tasks': sub,
                    'demand': sum(params['q'][t] for t in sub),
                    'service_time': sum(params['tau'][t] for t in sub)
                })
            continue

        # 级别二: 负载均衡 (最小化预计完成时间)
        min_finish_time = float('inf')
        for r_idx in feasible_robots:
            robot = robot_states[r_idx]
            # 估算新增时间和距离
            # 简单起见，假设从机器人最后一个任务点到簇中最近的任务点
            last_pos = robot['last_pos']
            cluster_tasks = cluster['tasks']
            
            # 找到簇中最近的入口任务
            entry_task = min(cluster_tasks, key=lambda t: params['dist'][(last_pos, t)])
            dist_to_cluster = params['dist'][(last_pos, entry_task)]
            
            # 估算簇内路径 (简单NN)
            internal_dist, internal_time = 0, 0
            unvisited = set(cluster_tasks)
            current = entry_task
            unvisited.remove(current)
            while unvisited:
                # ε-贪心策略：10%的概率从前2-3名中随机选择
                EPS_GREEDY = 0.1
                cand = list(unvisited)
                cand.sort(key=lambda t: params['dist'][(current, t)])
                if random.random() < EPS_GREEDY and len(cand) > 1:
                    next_task = cand[random.randint(0, min(2, len(cand)-1))]  # 从前2~3名里随机
                else:
                    next_task = cand[0]
                
                d = params['dist'][(current, next_task)]
                internal_dist += d
                internal_time += d / params['v_base'][r_idx] + params['tau'][next_task]
                current = next_task
                unvisited.remove(current)

            # 预计的新完成时间
            new_time = robot['time'] + (dist_to_cluster / params['v_base'][r_idx]) + internal_time
            
            if new_time < min_finish_time:
                min_finish_time = new_time
                candidate_robots = [(r_idx, new_time, dist_to_cluster + internal_dist)]
            elif new_time == min_finish_time:
                candidate_robots.append((r_idx, new_time, dist_to_cluster + internal_dist))
        
        # 级别三: 效率 (最小化新增行驶距离)
        if len(candidate_robots) > 1:
            min_dist = min(c[2] for c in candidate_robots)
            candidate_robots = [c for c in candidate_robots if c[2] == min_dist]

        # 级别四: 异构性 (能效更高)
        if len(candidate_robots) > 1:
            indices = [c[0] for c in candidate_robots]
            min_ef = min(params['ef'][i+1] for i in indices)
            best_indices = [i for i in indices if params['ef'][i+1] == min_ef]
            candidate_robots = [c for c in candidate_robots if c[0] in best_indices]

        # 级别五: 随机性
        best_robot_idx = random.choice(candidate_robots)[0]
        
        # 分配并更新机器人状态
        chosen_robot = robot_states[best_robot_idx]
        chosen_robot['tasks'].extend(cluster['tasks'])
        chosen_robot['load'] += cluster['demand']
        # 粗略更新时间和位置，以用于下一轮决策
        chosen_robot['time'] = min_finish_time 
        # 找到簇内离depot最远的点作为下一个出发点（粗略估计）
        chosen_robot['last_pos'] = max(cluster['tasks'], key=lambda t: params['dist'][(0, t)])


    # ================== 阶段三: 路径规划与最终解生成 (注入贪婪逻辑) ==================
    
    predecessors = {i: set() for i in range(1, params['M'] + 1)}
    for u, v in params['P']:
        predecessors[v].add(u)

    task_to_robot_map = {}
    for r_idx, r_state in enumerate(robot_states):
        for task in r_state['tasks']:
            task_to_robot_map[task] = r_idx + 1 # robot_id is 1-based

    all_tasks_in_clusters = set(task_to_robot_map.keys())
    unvisited_tasks = all_tasks_in_clusters.copy()
    completed_tasks = set()
    final_permutation = []

    # 追踪每个机器人最后一个任务的位置，以便计算最近邻
    # 初始位置都在仓库 (0)
    last_pos_per_robot = {k: 0 for k in range(1, params['K'] + 1)}

    # 3.2 循环构建高质量排列
    while unvisited_tasks:
        # 找到当前所有可执行的任务（其前序任务都已完成）
        candidate_tasks = [
            t for t in unvisited_tasks 
            if predecessors.get(t, set()).issubset(completed_tasks)
        ]

        if not candidate_tasks:
            # 如果没有候选任务，但仍有未访问任务，说明存在无法满足的约束（或bug）
            # 此时跳出，让后续的随机填充逻辑来处理剩余任务
            print(f"[警告] 初始化时发现无法满足的先后约束，剩余 {len(unvisited_tasks)} 个任务将随机填充。")
            break

        # 从候选任务中，根据贪婪的最近邻原则选择最佳任务（带随机性）
        cand_d = []
        for task in candidate_tasks:
            robot_id = task_to_robot_map.get(task)
            if robot_id is None: continue # 安全检查

            last_pos = last_pos_per_robot[robot_id]
            dist = params['dist'][(last_pos, task)]
            cand_d.append((dist, task))
        
        if not cand_d:
            # 这种情况理论上不应该发生，除非候选任务都不在 task_to_robot_map 中
            # 作为安全措施，随机选一个
            best_task = random.choice(candidate_tasks)
        else:
            # 从前K个最近的任务中随机选择，增加多样性
            cand_d.sort(key=lambda x: x[0])
            PICK_K = 3
            k = min(PICK_K, len(cand_d))
            best_task = random.choice([t for _, t in cand_d[:k]])

        # 3.3 更新状态
        final_permutation.append(best_task)
        unvisited_tasks.remove(best_task)
        completed_tasks.add(best_task)
        
        # 更新执行了该任务的机器人的最后位置
        chosen_robot_id = task_to_robot_map[best_task]
        last_pos_per_robot[chosen_robot_id] = best_task
    
    # 确保所有任务都被分配
    assigned_tasks = set(final_permutation)
    all_tasks_set = set(range(1, params['M'] + 1))
    if assigned_tasks != all_tasks_set:
        # 将聚类中未包含的任务 和 循环中未处理的任务 一并随机填充
        unassigned = list(all_tasks_set - assigned_tasks)
        random.shuffle(unassigned)
        final_permutation.extend(unassigned)

    return np.array(final_permutation) - 1



def stochastic_universal_sampling(population_indices, rank_map, crowding_dists, num_selections):
   
    n = len(list(population_indices))
    if n == 0:
        return []
    
    def _crowd_key(i):
        rank = rank_map.get(i, n + 1)
        crowding = crowding_dists[i]
        # 拥挤度越大越好 (排序键越小越好)，因此无限大的拥挤度应该对应一个极小的值
        if np.isinf(crowding):
            return (rank, -1e30) # 用一个极小的负数代表无限大的拥挤度
        return (rank, -crowding)

    sorted_indices = sorted(population_indices, key=_crowd_key)

   
    fitness_map = {idx: n - i for i, idx in enumerate(sorted_indices)}
    total_fitness = sum(fitness_map.values())

    if total_fitness == 0:
        # 极端情况的备用策略: 如果所有适应度为0，则随机选择
        return random.choices(sorted_indices, k=num_selections)
   
    selected_parents = []
    step = total_fitness / num_selections
    start_ptr = random.uniform(0, step)
    
    cumulative_fitness = 0
    current_idx_in_sorted = 0 # 指向 sorted_indices 的指针
    
    for i in range(num_selections):
        selection_ptr = start_ptr + i * step
        
        # 移动指针，直到找到包含当前选择指针的适应度区间
        while cumulative_fitness < selection_ptr:
            individual_index = sorted_indices[current_idx_in_sorted]
            cumulative_fitness += fitness_map[individual_index]
            current_idx_in_sorted += 1
            if current_idx_in_sorted >= n:
                # 防止索引越界
                current_idx_in_sorted = n
                break
        
        # 指针落入的区域对应的个体就是被选中的父代
        selected_parent_index = sorted_indices[max(0, current_idx_in_sorted - 1)]
        selected_parents.append(selected_parent_index)
        
    return selected_parents

def update_archive(archive_solutions, archive_objectives, candidates_solutions, candidates_objectives, max_size):
   
    if not candidates_solutions:
        return archive_solutions, archive_objectives

    cand_obj_arr = np.array(candidates_objectives)

    # 使用形状检查更健壮 (archive_objectives 应该是 (0, 3) 或 (n, 3))
    if archive_objectives.shape[0] == 0:
        combined_solutions = candidates_solutions
        combined_objectives = cand_obj_arr
    else:
        combined_solutions = archive_solutions + candidates_solutions
        combined_objectives = np.vstack((archive_objectives, cand_obj_arr))

    # 找到非支配前沿
    fronts, _ = robust_non_dominated_sorting(combined_objectives)
    pareto_indices = fronts[0] if fronts and fronts[0] else []
    
    archive_solutions = [combined_solutions[i] for i in pareto_indices]
    archive_objectives = combined_objectives[pareto_indices]

    # 剪枝
    if len(archive_solutions) > max_size:
        crowding_dists = calculate_crowding_distance(archive_objectives, [list(range(len(archive_solutions)))])
        sorted_indices = np.argsort(crowding_dists)[::-1] # 拥挤度大的优先
        pruned_indices = sorted_indices[:max_size]
        
        archive_solutions = [archive_solutions[i] for i in pruned_indices]
        archive_objectives = archive_objectives[pruned_indices]
        
    return archive_solutions, archive_objectives



def get_dominance_reward(obj_old, obj_new, archive_objectives=None):
    
    obj_old, obj_new = np.array(obj_old), np.array(obj_new)

    if np.array_equal(obj_old, obj_new):
        return 0.0

    if np.all(obj_new <= obj_old) and np.any(obj_new < obj_old):
        return 1.0  # 恢复：将支配奖励调整为1.0，以恢复之前强大的收敛性

    if np.all(obj_old <= obj_new) and np.any(obj_old < obj_new):
        return -1.0 # Strong penalty for moving backwards

    # Mutually non-dominating case: reward based on diversity
    if archive_objectives is None or archive_objectives.shape[0] == 0:
        return 0.5 # Fallback to original reward if archive is empty

    # Calculate distance to nearest neighbor in the archive
    # Reshape obj_new to (1, 3) to ensure correct broadcasting
    distances = np.linalg.norm(archive_objectives - obj_new.reshape(1, -1), axis=1)
    min_dist = np.min(distances)

    # Normalize the distance. Use the diagonal of the objective space bounding box of the archive.
    if archive_objectives.shape[0] > 1:
        obj_ranges = np.max(archive_objectives, axis=0) - np.min(archive_objectives, axis=0)
        obj_ranges[obj_ranges == 0] = 1.0 # Prevent division by zero
        norm_factor = np.sqrt(np.sum(obj_ranges**2))
        normalized_dist = min_dist / max(norm_factor, 1e-6)
    else:
        # Fallback for archives with a single point
        norm_factor = np.linalg.norm(archive_objectives[0])
        normalized_dist = min_dist / max(norm_factor, 1e-6)

    # Reward is a combination of a base value and the diversity bonus (capped at 1.0)
    diversity_reward = 0.5 + 1.0 * min(normalized_dist, 1.0) # 增强多样性奖励: 基础奖励0.5，多样性部分最高1.0，总奖励可达1.5

    return diversity_reward


def get_individual_features_from_routes(routes, params):
    
    route_lengths = []
    all_route_durations = []  # 修改：包含所有机器人的完成时间
    num_routes_used = 0

    # 修改：遍历所有机器人，不只是有任务的
    for r in range(1, params['K'] + 1):
        tasks = routes.get(r, [])
        if tasks:
            num_routes_used += 1
            route_lengths.append(len(tasks))
            
            # 计算路线时长 (与主目标函数逻辑一致)
            current_completion_time = 0
            path = [0] + tasks + [0]
            for i in range(len(path) - 1):
                start_node, end_node = path[i], path[i+1]
                travel_time = params['dist'][(start_node, end_node)] / params['v_base'][r-1]
                service_start_time = current_completion_time + travel_time
                if end_node != 0:
                    current_completion_time = service_start_time + params['tau'][end_node]
                else:
                    current_completion_time = service_start_time
            all_route_durations.append(current_completion_time)
        else:
            # 未使用的机器人完成时间为0
            all_route_durations.append(0.0)

    # 修改：与Gurobi一致，考虑所有机器人计算不均衡
    imbalance = max(all_route_durations) - min(all_route_durations) if all_route_durations else 0
    max_route_length = max(route_lengths) if route_lengths else 0
    avg_route_length = np.mean(route_lengths) if route_lengths else 0

    return {
        'imbalance': imbalance,
        'max_route_length': max_route_length,
        'avg_route_length': avg_route_length,
        'num_routes_used': num_routes_used
    }

def repair_routes(routes, params):
    
    from collections import Counter
    
    all_tasks_set = set(range(1, params['M'] + 1))
    
    # 1. 统计所有路径中的任务，并识别重复项和缺失项
    flat_tasks = [task for r in sorted(routes.keys()) for task in routes.get(r, [])]
    task_counts = Counter(flat_tasks)
    
    missing_tasks = list(all_tasks_set - set(task_counts.keys()))
    duplicate_tasks = [task for task, count in task_counts.items() if count > 1]
    
    # 2. 从路径中移除重复的任务实例
    if duplicate_tasks:
        seen_once = set()
        for r in sorted(routes.keys()):
            new_path = []
            for task in routes.get(r, []):
                # 如果是重复任务
                if task in duplicate_tasks:
                    if task not in seen_once:
                        new_path.append(task)
                        seen_once.add(task)
                    else: # 这是重复的实例，将其加入待办列表
                        missing_tasks.append(task)
                # 如果不是重复任务
                else:
                    new_path.append(task)
            routes[r] = new_path

    # 3. 使用贪心策略重新插入所有缺失和被移除的任务
    # 为了提高效率，按需求量从大到小插入，这是一种启发式策略
    missing_tasks.sort(key=lambda t: params['q'][t], reverse=True)

    for task in missing_tasks:
        best_insertion = None # (min_cost, robot_id, position)
        
        # 遍历所有机器人和所有可能的插入位置
        for r_id in range(1, params['K'] + 1):
            # 检查容量约束
            current_load = sum(params['q'][t] for t in routes.get(r_id, []))
            if current_load + params['q'][task] > params['Q'][r_id-1]:
                continue # 容量不足，跳过此机器人
            
            # 评估插入到每个位置的成本（这里简化为距离增量）
            path = routes.get(r_id, [])
            for i in range(len(path) + 1):
                prev_node = path[i-1] if i > 0 else 0
                next_node = path[i] if i < len(path) else 0
                
                # 计算插入成本: (d_prev,task + d_task,next) - d_prev,next
                time_term = (params['dist'].get((prev_node, task), 1e9) / params['v_base'][r_id-1] + params['tau'][task])
                cost = (params['dist'].get((prev_node, task), 1e9)
                        + params['dist'].get((task, next_node), 1e9)
                        - params['dist'].get((prev_node, next_node), 1e9)) + 0.5 * time_term
                
                if best_insertion is None or cost < best_insertion[0]:
                    best_insertion = (cost, r_id, i)

        # 执行最佳插入
        if best_insertion:
            _, r_id, pos = best_insertion
            if r_id not in routes: routes[r_id] = []
            routes[r_id].insert(pos, task)
        else:
            # 如果没有找到任何可行的插入点（极为罕见，除非单个任务就超重）
            # 备用策略: 强制放入剩余容量最大的机器人路径末尾
            best_robot = -1
            max_rem_cap = -1
            for r in range(1, params['K'] + 1):
                rem_cap = params['Q'][r-1] - sum(params['q'][t] for t in routes.get(r, []))
                if rem_cap > max_rem_cap:
                    max_rem_cap = rem_cap
                    best_robot = r
            
            if best_robot != -1:
                if best_robot not in routes: routes[best_robot] = []
                routes[best_robot].append(task)
            else:
                 # 连备用策略都失败，这是最极端的情况，随机分配
                r_id = random.randint(1, params['K'])
                if r_id not in routes: routes[r_id] = []
                routes[r_id].append(task)

    return routes

def repair_precedence(routes, params, max_pass=2):
   
    def index_map_of(route): return {t:i for i,t in enumerate(route)}

    for _ in range(max_pass):
        changed = False
        s, f = simulate_task_times(routes, params)

        belong = {}
        for r_id, path in routes.items():
            for t in path:
                belong[t] = r_id

        for (u, v) in params['P']:
            ru, rv = belong.get(u), belong.get(v)

            # 同一路径且次序颠倒
            if ru is not None and rv is not None and ru == rv:
                idx = index_map_of(routes[ru])
                if idx.get(v, 1e9) < idx.get(u, -1e9):
                    routes[ru].pop(idx[v])
                    idx = index_map_of(routes[ru])
                    routes[ru].insert(idx[u] + 1, v)
                    changed = True
                    continue

            # 跨路径违约：f[u] > s[v]
            if u in f and v in s and f[u] > s[v]:
                if rv is not None and v in routes[rv]:
                    # 先尝试在原路径把 v 往后简单挪到末尾
                    try:
                        routes[rv].remove(v)
                        routes[rv].append(v)
                        changed = True
                    except ValueError:
                        pass

                # 再算一次，仍违约就把 v 迁到 u 的路径，插在 u 之后的最优位置（看距离增量 + 容量）
                s2, f2 = simulate_task_times(routes, params)
                if u in f2 and v in s2 and f2[u] > s2[v]:
                    if ru is not None:
                        cur_load = sum(params['q'][t] for t in routes[ru])
                        if cur_load + params['q'][v] <= params['Q'][ru-1]:
                            idx_u = index_map_of(routes[ru])[u]
                            best = None
                            for pos in range(idx_u+1, len(routes[ru])+1):
                                prev_node = routes[ru][pos-1] if pos>0 else 0
                                next_node = routes[ru][pos] if pos<len(routes[ru]) else 0
                                delta = (params['dist'][(prev_node, v)]
                                         + params['dist'][(v, next_node)]
                                         - params['dist'][(prev_node, next_node)])
                                if best is None or delta < best[0]:
                                    best = (delta, pos)
                            # 从原路径移除 v
                            if rv is not None and v in routes[rv]:
                                routes[rv].remove(v)
                            routes[ru].insert(best[1], v)
                            changed = True
        if not changed:
            break
    return routes


def solve_instance(problem_file_path, show_plot=False):
    
    print(f"\n{'='*25} NEW INSTANCE {'='*25}")
    print(f"正在从 '{problem_file_path}' 加载问题数据...")
    
    try:
        params = load_problem_data(problem_file_path)
    except Exception as e:
        print(f"[严重错误] 加载问题 '{os.path.basename(problem_file_path)}' 失败: {e}")
        return # 跳过此算例

    
    TIME_LIMIT_SECONDS = params['M'] * params['K'] * 1.0 # M * K (s)
    print(f"根据任务数(M={params['M']})和机器人数(K={params['K']})，动态计算出的终止时间上限为: {TIME_LIMIT_SECONDS:.2f} 秒")
    
    POP_SIZE = 140
    
    # ============= 并行评估参数 =============
    N_JOBS = 1
    PARALLEL_BACKEND = "threading"
    # ======================================
    
    N_GEN = 20000
        
    WARMUP_GENERATIONS = 150
    DQN_LEARN_PERIOD = 1

    # --- 新增: 精英档案库参数 ---
    ARCHIVE_SIZE = 100
    ARCHIVE_INJECT_PERIOD = 25
    ARCHIVE_INJECT_COUNT = 5
    # --------------------------

    # DQN 超参数
    DQN_HYPARAMS = {
        'BATCH_SIZE': 128, 'GAMMA': 0.99, 'EPS_START': 0.9,
        'EPS_END': 0.15, 'TAU': 0.005, 'LR': 1e-4, 'MEMORY_SIZE': 10000
    }
    N_OBSERVATIONS = 7
    N_ACTIONS = 5
    
    # --- 动态调整探索衰减率 ---
    eps_decay_dynamic = int((params['M'] * params['K']) * 125)
    DQN_HYPARAMS['EPS_DECAY'] = eps_decay_dynamic
    print(f"[DQN配置] 动态计算出的探索衰减步数为: {eps_decay_dynamic}")
    # ------------------------------------

    agent = DQNAgent(N_OBSERVATIONS, N_ACTIONS, DQN_HYPARAMS)
    agent.action_history = []
    
    action_map = {
        0: als.critical_task_push, 1: als.coordinated_cluster_transfer,
        2: als.route_ejection_greedy_absorption, 3: als.multi_route_synergy_refinement,
        4: als.large_scale_destruction_reconstruction
    }
    
    action_stats = {i: 0 for i in range(N_ACTIONS)} if USE_DQN else None

    print(f"初始化种群 ({POP_SIZE} 个体)，使用多阶段启发式策略...")
    params['_debug_decode'] = params['M'] <= 50
    
    print("  (1/2) 生成高质量的初始排列...")
    initial_permutations = [generate_heuristic_individual(params, None) for i in range(POP_SIZE)]
    print("  (2/2) 将排列解码为路径，构建初始种群...")
    population = [decode_chromosome((p + 1).tolist(), params) for p in initial_permutations]
    
    objectives = evaluate_population(population, params, n_jobs=N_JOBS, backend=PARALLEL_BACKEND)

    archive_solutions = []
    archive_objectives = np.empty((0, 3), dtype=float)

    print("开始运行 Memetic NSGA-II (V4 - Direct Route Representation)...")
    print("\nEvolution started...")
    start_time = time.time()
    
    stop_reason = None
    
    try:
        for gen in range(N_GEN):
            elapsed_time = time.time() - start_time
            if elapsed_time > TIME_LIMIT_SECONDS:
                stop_reason = f"达到墙钟时间上限 ({elapsed_time:.2f}s / {TIME_LIMIT_SECONDS:.2f}s)"
                print(f"\n{stop_reason}。正在终止进化...")
                break
            
            gen_start_time = time.time()  # 记录每代开始时间
            
            # 设置当前代数用于解码器输出控制
            params['current_generation'] = gen
            
            # 检查是否超时
            elapsed_time = time.time() - start_time
            if elapsed_time > TIME_LIMIT_SECONDS:
                stop_reason = f"达到墙钟时间上限 ({elapsed_time:.2f}s / {TIME_LIMIT_SECONDS:.2f}s)"
                print(f"\n{stop_reason}。正在终止进化...")
                break
            
            # 记录种群大小
            # population_sizes.append(len(population))

            # ==============================================================================================
            # 步骤 1: 评估父代，为选择做准备
            # ==============================================================================================
            step_start = time.time()
            objectives_array = np.array(objectives)
            current_fronts, rank_map = robust_non_dominated_sorting(objectives_array)
            current_crowding_dists = calculate_crowding_distance(objectives_array, current_fronts)
            ranking_time = time.time() - step_start
            
            # ==============================================================================================
            # 步骤 2: 生成子代 (通过模因操作 - DQN指导的局部搜索)
            # ==============================================================================================
            memetic_op_start = time.time()
            offspring = []
            experiences_to_process = []
            is_warmup = gen < WARMUP_GENERATIONS # 恢复为固定的代数判断

            # --- 状态归一化因子 (全局固定值) ---
            MAX_IMBALANCE = params['M'] * 100 
            MAX_ROUTE_LENGTH = params['M'] 
            MAX_AVG_LENGTH = params['M'] / params['K'] if params['K'] > 0 else params['M']
            MAX_NUM_ROUTES = params['K']

            # 防止除零
            MAX_IMBALANCE = max(MAX_IMBALANCE, 1.0)
            MAX_ROUTE_LENGTH = max(MAX_ROUTE_LENGTH, 1.0)
            MAX_AVG_LENGTH = max(MAX_AVG_LENGTH, 1.0)
            MAX_NUM_ROUTES = max(MAX_NUM_ROUTES, 1.0)

            # --- DQN状态所需的全局信息 ---
            max_rank = max(rank_map.values()) if rank_map else 0
            crowding_vals = current_crowding_dists[np.isfinite(current_crowding_dists)]
            max_crowding = np.max(crowding_vals) if len(crowding_vals) > 0 else 1.0
            max_crowding = max(max_crowding, 1.0)

            # --- 新增: 精英档案库注入 ---
            if gen > 0 and gen % ARCHIVE_INJECT_PERIOD == 0 and archive_solutions:
                # 识别要替换的种群中最差的个体
                # 排序键: rank(越高越差), 拥挤度(越低越差)
                pop_metrics = []
                for i in range(len(population)):
                    crowding = current_crowding_dists[i]
                    if not np.isfinite(crowding):
                        crowding = 1e9 # for sorting, treat inf as very high crowding (good)
                    pop_metrics.append((rank_map[i], crowding, i))
                
                # 按 rank 升序, 拥挤度降序, 得到从优到劣的排序
                pop_metrics.sort(key=lambda x: (x[0], -x[1])) 
                
                # 最差的个体在列表末尾
                indices_to_replace = [metric[2] for metric in pop_metrics[-ARCHIVE_INJECT_COUNT:]]
                
                # 从档案库中随机选择要注入的解
                num_to_inject = min(ARCHIVE_INJECT_COUNT, len(archive_solutions), len(indices_to_replace))
                solutions_to_inject_indices = random.sample(range(len(archive_solutions)), num_to_inject)
                
                replaced_count = 0
                for i in range(num_to_inject):
                    replace_idx = indices_to_replace[i]
                    archive_sol_idx = solutions_to_inject_indices[i]
                    
                    # 执行替换
                    population[replace_idx] = copy.deepcopy(archive_solutions[archive_sol_idx])
                    objectives[replace_idx] = archive_objectives[archive_sol_idx].tolist()
                    replaced_count += 1

                if replaced_count > 0:
                    # 注入后，必须重新评估排名和拥挤度
                    objectives_array = np.array(objectives)
                    current_fronts, rank_map = robust_non_dominated_sorting(objectives_array)
                    current_crowding_dists = calculate_crowding_distance(objectives_array, current_fronts)
                    
                    # 由于排名和拥挤度变化，DQN状态归一化因子也需要重新计算
                    max_rank = max(rank_map.values()) if rank_map else 0
                    crowding_vals = current_crowding_dists[np.isfinite(current_crowding_dists)]
                    max_crowding = np.max(crowding_vals) if len(crowding_vals) > 0 else 1.0
                    max_crowding = max(max_crowding, 1.0)
            # ---------------------------

            # --- 生成与父代种群大小相同的子代 ---
            # 1. 使用高级选择算子选择所有父代
            parent_indices = stochastic_universal_sampling(
                population_indices=range(len(population)), 
                rank_map=rank_map, 
                crowding_dists=current_crowding_dists, 
                num_selections=POP_SIZE
            )

            current_gen_actions = [] # 为当前代创建一个动作记录列表

            for parent_idx in parent_indices:
                # 从预选的父代列表中获取父代
                parent_routes = population[parent_idx]
                parent_obj = objectives[parent_idx]
                
                child_routes, child_obj = None, None
                
                if is_warmup:
                    # 在预热阶段，使用随机一个高级算子作为变异
                    # 恢复均匀采样，让DQN充分学习所有算子的效果
                    action_idx = random.randrange(N_ACTIONS)
                    action_func = action_map[action_idx]
                    try:
                        child_routes, _ = action_func(copy.deepcopy(parent_routes), params)
                        child_routes = repair_routes(child_routes, params)
                        child_routes = repair_precedence(child_routes, params)  # ← 新增
                        child_obj = evaluate(child_routes, params)
                    except Exception:
                        child_routes = copy.deepcopy(parent_routes)
                        child_obj = parent_obj
                else:
                    # 在模因阶段，使用DQN指导的局部搜索
                    # 2. 构建父代状态 - 升级为9维
                    parent_rank = rank_map[parent_idx]
                    parent_crowding = current_crowding_dists[parent_idx]
                    parent_features = get_individual_features_from_routes(parent_routes, params)
                    
                    rank_norm = parent_rank / max(1, max_rank)
                    raw = parent_crowding
                    crowding_norm = (raw / max_crowding) if np.isfinite(raw) else 1.0  # 边界点给 1.0
                    stage = float(gen) / N_GEN
                    imbalance_norm = parent_features['imbalance'] / MAX_IMBALANCE
                    max_len_norm = parent_features['max_route_length'] / MAX_ROUTE_LENGTH
                    avg_len_norm = parent_features['avg_route_length'] / MAX_AVG_LENGTH
                    num_routes_norm = parent_features['num_routes_used'] / MAX_NUM_ROUTES
                    
                    state = torch.tensor([[
                        rank_norm, crowding_norm, stage,
                        imbalance_norm, max_len_norm, avg_len_norm, num_routes_norm
                    ]], device=device, dtype=torch.float)
                    
                    # 3. 算子选择：DQN智能选择 vs 随机选择（对比实验）
                    if USE_DQN:
                        action_idx = agent.select_action(state).item()  # DQN智能选择
                        action_stats[action_idx] += 1  # 统计动作选择
                        current_gen_actions.append(action_idx) # <-- 在这里记录动作
                    else:
                        action_idx = random.randint(0, len(action_map) - 1)  # 随机选择作为基准
                        current_gen_actions.append(action_idx) # <-- 在这里也记录动作
                    action_func = action_map[action_idx]
                    
                    # 4. 执行局部搜索生成子代
                    try:
                        child_routes, _ = action_func(copy.deepcopy(parent_routes), params)
                        # --- 核心优化: 在评估前修复解 ---
                        child_routes = repair_routes(child_routes, params)
                        child_routes = repair_precedence(child_routes, params)  # ← 新增
                        child_obj = evaluate(child_routes, params)
                    except Exception as e:
                        # 如果算子失败，子代就是父代的克隆
                        child_routes = copy.deepcopy(parent_routes)
                        child_obj = parent_obj
                        
                    # 5. 计算奖励并准备学习经验
                    reward = get_dominance_reward(parent_obj, child_obj, archive_objectives)
                    
                    # 6. 【核心改造】构建真实的 next_state，让 GAMMA 生效
                    # 我们用子代的新特征 + 父代的种群特征来构成下一状态
                    child_features = get_individual_features_from_routes(child_routes, params)
                    
                    # 使用与父代相同的归一化因子
                    imbalance_norm_next = child_features['imbalance'] / MAX_IMBALANCE
                    max_len_norm_next = child_features['max_route_length'] / MAX_ROUTE_LENGTH
                    avg_len_norm_next = child_features['avg_route_length'] / MAX_AVG_LENGTH
                    num_routes_norm_next = child_features['num_routes_used'] / MAX_NUM_ROUTES
                    
                    # 复用父代的rank, crowding和当前的stage
                    next_state = torch.tensor([[
                        rank_norm, crowding_norm, stage,
                        imbalance_norm_next, max_len_norm_next, avg_len_norm_next, num_routes_norm_next
                    ]], device=device, dtype=torch.float)

                    experiences_to_process.append({
                        "state": state,
                        "action": torch.tensor([[action_idx]], device=device, dtype=torch.long),
                        "next_state": next_state, # <-- 使用新构建的 next_state
                        "reward": torch.tensor([reward], device=device)
                    })
                
                # 7. 添加子代到新种群
                offspring.append(child_routes)

            # 将当前代的动作历史存入总历史
            if hasattr(agent, 'action_history'):
                # 将动作与当前经过的时间戳 (毫秒) 一同记录
                agent.action_history.append({
                    "time_ms": elapsed_time * 1000,
                    "actions": current_gen_actions
                })

            # --- 步骤 2.5: 评估所有新生成的子代 ---
            offspring_objectives = evaluate_population(offspring, params, n_jobs=N_JOBS, backend=PARALLEL_BACKEND)
            
            # --- 步骤 3: DQN 学习与更新 (已整合) ---
            if not is_warmup:
                # 3.1 将本轮经验存入回放池
                for exp in experiences_to_process:
                    agent.memory.push(exp['state'], exp['action'], exp['next_state'], exp['reward'])
                
                # 3.2 定期训练DQN模型并更新目标网络
                # 条件: 1) 已过预热期 2) 到达学习周期 3) 经验池大小足够
                if gen % DQN_LEARN_PERIOD == 0 and len(agent.memory) > DQN_HYPARAMS['BATCH_SIZE']:
                    agent.learn()
                    agent.update_target_net()

            # --- 步骤 4: 环境选择 (精英保留) ---
            combined_pop = population + offspring
            combined_obj = np.array(objectives + offspring_objectives)

            # 健康检查: 确保没有无效的目标值
            invalid_mask = ~np.all(np.isfinite(combined_obj), axis=1)
            if np.any(invalid_mask):
                print(f"\n[警告] 在第 {gen+1} 代发现 {np.sum(invalid_mask)} 个无效目标值。已替换为惩罚值。")
                combined_obj[invalid_mask] = np.array([1e9, 1e9, 1e9])
            
            # 计算合并种群的非支配前沿和拥挤度
            all_fronts, _ = robust_non_dominated_sorting(combined_obj)
            crowding_dists = calculate_crowding_distance(combined_obj, all_fronts)

            # 精英保留策略选出下一代
            next_population = []
            next_objectives = []
            last_front_idx = -1

            for i, front in enumerate(all_fronts):
                if len(next_population) + len(front) <= POP_SIZE:
                    for j in front:
                        next_population.append(combined_pop[j])
                        next_objectives.append(combined_obj[j])
                else:
                    last_front_idx = i
                    break

            if last_front_idx != -1 and len(next_population) < POP_SIZE:
                last_front = all_fronts[last_front_idx]
                remaining = POP_SIZE - len(next_population)

                # 根据拥挤度降序排序
                order = sorted(last_front, key=lambda j: crowding_dists[j], reverse=True)
                for j in order[:remaining]:
                    next_population.append(combined_pop[j])
                    next_objectives.append(combined_obj[j])

            # 更新种群
            population = next_population
            objectives = next_objectives
            
            # --- 新增: 更新精英档案库 ---
            # 使用新一代种群的帕累托前沿来更新档案库
            current_objectives_np = np.array(objectives)
            if current_objectives_np.size > 0:
                current_fronts, _ = robust_non_dominated_sorting(current_objectives_np)
                if current_fronts and current_fronts[0]:
                    pareto_front_indices = current_fronts[0]
                    pareto_solutions = [population[i] for i in pareto_front_indices]
                    pareto_objectives = current_objectives_np[pareto_front_indices]
            
                    archive_solutions, archive_objectives = update_archive(
                        archive_solutions, archive_objectives,
                        pareto_solutions, pareto_objectives,
                        ARCHIVE_SIZE
                    )
                    # 档案库已更新
            # --------------------------

            if not population:
                print("[警告] 新一代种群为空，可能所有解都无效。正在重新初始化...")
                initial_perms = [generate_heuristic_individual(params, None) for _ in range(POP_SIZE)]
                population = [decode_chromosome((p + 1).tolist(), params) for p in initial_perms]
                objectives = evaluate_population(population, params, n_jobs=N_JOBS, backend=PARALLEL_BACKEND)
            
            # 打印帕累托前沿大小 (使用上面已计算好的current_fronts)
            pareto_front_size = len(current_fronts[0]) if current_fronts and len(current_fronts[0]) > 0 else 0

            # 记录每代的性能信息
            gen_end_time = time.time()
            gen_duration = gen_end_time - gen_start_time
            # generation_times.append(gen_duration)
            
            if (gen + 1) % 10 == 0:
                print(f"代: {gen+1}/{N_GEN}, Pareto前沿: {pareto_front_size}")

        if stop_reason is None: # 如果循环正常结束
            last_gen = locals().get('gen', -1) + 1  # 容错: 如果gen未定义则为0
            stop_reason = f"达到最大迭代次数 (gen={last_gen})"

    except Exception as e:
        print("\n!!!!!! An unexpected error occurred during the evolution loop !!!!!!")
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Details: {e}")
        import traceback
        traceback.print_exc()
        stop_reason = f"异常退出: {type(e).__name__}" # <-- 记录异常为停止原因

    finally:
        total_runtime = time.time() - start_time
        print(f"\n[STOP] 停止原因: {stop_reason}, 总运行时间: {total_runtime:.2f}s, 设定上限: {TIME_LIMIT_SECONDS:.2f}s")
        print("Evolution finished or stopped. Saving final results...")
        
        # ======================= [ 结果汇总：提取 Pareto 前沿 ] =======================
        final_pop_obj = np.array(evaluate_population(population, params, n_jobs=N_JOBS, backend=PARALLEL_BACKEND))
        final_pop_solutions = population # 个体已经是 routes

        fronts, _ = robust_non_dominated_sorting(final_pop_obj)
        pareto_front_indices = fronts[0] if fronts and fronts[0] else []

        pareto_front_objectives = []
        pareto_front_solutions = []

        # 完全移除去重 - 保留所有Pareto最优解
        total_pareto_solutions = len(pareto_front_indices)
        
        for i in pareto_front_indices:
            obj = final_pop_obj[i]
            pareto_front_objectives.append(obj.tolist())
            pareto_front_solutions.append(final_pop_solutions[i])
        
        # 多样性诊断
        print(f"[多样性诊断] 保留所有{total_pareto_solutions}个Pareto最优解（无去重）")
        
        # 分析目标值的多样性和质量
        if total_pareto_solutions > 1:
            obj_array = np.array(pareto_front_objectives)
            dist_range = obj_array[:, 0].max() - obj_array[:, 0].min()
            energy_range = obj_array[:, 1].max() - obj_array[:, 1].min()
            imbalance_range = obj_array[:, 2].max() - obj_array[:, 2].min()
            
            # 质量分析
            avg_dist = obj_array[:, 0].mean()
            avg_energy = obj_array[:, 1].mean()
            avg_imbalance = obj_array[:, 2].mean()
            
            print(f"[多样性分析] 距离范围: {dist_range:.2f}, 能耗范围: {energy_range:.2f}, 不均衡范围: {imbalance_range:.2f}")
            print(f"[质量分析] 平均距离: {avg_dist:.2f}, 平均能耗: {avg_energy:.2f}, 平均不均衡: {avg_imbalance:.2f}")
            
            # 检查是否目标值异常大
            task_per_robot = params['M'] / params['K']
            expected_avg_dist = task_per_robot * 50  # 粗略估计：每个任务间距50
            if avg_dist > expected_avg_dist * 2:
                print(f"[警告] 平均距离({avg_dist:.2f})异常大，可能路径规划有问题。预期约{expected_avg_dist:.2f}")
        elif total_pareto_solutions == 1:
            obj = pareto_front_objectives[0]
            print(f"[质量分析] 单解目标值: 距离={obj[0]:.2f}, 能耗={obj[1]:.2f}, 不均衡={obj[2]:.2f}")

        # （可选）展示/签名阶段让解码器确定性
        params['decoder_noise'] = 0.0
        
        # ======================= [ 参数设置 ] =======================
        # 保存使用的关键参数
        params_to_save = {
            'POP_SIZE': POP_SIZE,
            'N_GEN': N_GEN,
            'WARMUP_GENERATIONS': WARMUP_GENERATIONS,
            'DQN_LEARN_PERIOD': DQN_LEARN_PERIOD,
            'ARCHIVE_SIZE': ARCHIVE_SIZE,
            'ARCHIVE_INJECT_PERIOD': ARCHIVE_INJECT_PERIOD
        }
        # =============================================================

        # ======================= [ 结果保存 ] =======================
        problem_name = os.path.splitext(os.path.basename(problem_file_path))[0]
        
        # 【重要】为结果添加解码后的路径和详细统计信息
        final_solutions_with_details = []
        for i, routes in enumerate(pareto_front_solutions):
            # 个体已经是 routes，不再需要解码
            robot_details = []
            # 遍历所有机器人以包含未使用的机器人
            for r_id in range(1, params['K'] + 1):
                tasks = routes.get(r_id, [])
                
                path_str = "0 -> " + " -> ".join(map(str, tasks)) + " -> 0" if tasks else "0 -> 0"
                total_demand = sum(params['q'][t] for t in tasks)
                capacity = params['Q'][r_id - 1]
                utilization = (total_demand / capacity) * 100 if capacity > 0 else 0
                
                robot_details.append({
                    "robot_id": r_id,
                    "path": path_str,
                    "tasks": tasks,
                    "total_demand": total_demand,
                    "capacity": capacity,
                    "utilization_percent": round(utilization, 2)
                })

            final_solutions_with_details.append({
                "solution_id": i + 1,
                "objectives": pareto_front_objectives[i],
                # "permutation": perm, # 不再有 permutation
                "robot_details": robot_details
            })

        res_data = {
            'problem_name': problem_name,
            'num_robots': params['K'],
            'num_tasks': params['M'],
            'runtime_s': total_runtime, # <-- 核心新增: 保存总运行时间
            'use_dqn': USE_DQN,  # 记录是否使用DQN
            'operator_selection_method': 'DQN' if USE_DQN else 'Random',  # 算子选择方法
            'pareto_front': final_solutions_with_details, # 使用新的、更详细的结构
            'solver_params': params_to_save
        }

        # 确保目录存在，根据是否使用DQN选择不同的输出目录
        # BUGFIX: 增加对__file__未定义环境（如notebook）的兼容性
        try:
            base_dir = os.path.dirname(__file__)
        except NameError:
            base_dir = os.getcwd()
            
        if USE_DQN:
            output_dir = os.path.join(base_dir, "results", "nsga2_v3_dqn")
        else:
            output_dir = os.path.join(base_dir, "results", "nsga2_v3_random")
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存结果
        result_filename = os.path.join(output_dir, f"{problem_name}_results.json")
        with open(result_filename, 'w') as f:
            json.dump(res_data, f, indent=4)
        mode_str = "DQN智能选择" if USE_DQN else "随机选择（基准）"
        print(f"NSGA-II ({mode_str}) 帕累托前沿数据已保存到 '{result_filename}'")
        
        # 输出DQN动作选择统计
        if USE_DQN and action_stats:
            print("\n" + "="*50)
            print("DQN动作选择统计:")
            action_names = [
                "critical_task_push",
                "coordinated_cluster_transfer", 
                "route_ejection_greedy_absorption",
                "multi_route_synergy_refinement",
                "large_scale_destruction_reconstruction"
            ]
            total_actions = sum(action_stats.values())
            for action_idx, count in action_stats.items():
                percentage = (count / total_actions * 100) if total_actions > 0 else 0
                print(f"  动作{action_idx} ({action_names[action_idx]}): {count}次 ({percentage:.1f}%)")
            print("="*50)
        
        # ======================= [ 详细结果打印 ] =======================
        print("\n" + "="*30 + " 求解完成：帕累托最优解详情 " + "="*30)
        if pareto_front_solutions:
            # 使用 numpy 数组进行排序
            pf_obj_np = np.array(pareto_front_objectives)
            sorted_indices = np.argsort(pf_obj_np[:, 0])
            
            print(f"找到 {len(pareto_front_solutions)} 个帕累托最优解。")
            
            for i in sorted_indices:
                solution = final_solutions_with_details[i]
                print(f"  解 {solution['solution_id']}: 总距离={solution['objectives'][0]:.2f}, 总能耗={solution['objectives'][1]:.2f}, 不均衡度={solution['objectives'][2]:.2f}")
                
                # 直接从结果中读取路径并打印
                robot_details = solution['robot_details']
                for detail in robot_details:
                    print(f"    - 机器人 {detail['robot_id']}: {detail['path']}")
        else:
            print("未能找到可行解。")
        print("="*78)

        # ======================= [ 可视化 ] =======================
        pf_obj_np = np.array(pareto_front_objectives)
        if show_plot and pf_obj_np.size > 0:
            try:
                from pymoo.visualization.scatter import Scatter
                last_gen = locals().get('gen', -1) + 1  # 容错: 如果gen未定义则为0
                plot = Scatter(title=f"Memetic NSGA-II Pareto Front (V4, Gen {last_gen})",
                               labels=["Distance", "Energy", "Imbalance"])
                plot.add(pf_obj_np, facecolor="red", edgecolor="red", s=40)
                plot.show()
            except ImportError:
                print("\n[警告] 未安装 'pymoo' 库，无法显示帕累托前沿图。请运行: pip install pymoo")
            except Exception as e:
                print(f"\n[警告] 生成帕累托前沿图时发生错误: {e}")
        elif show_plot:
            print("Could not generate plot because no solutions were found.")

        # +++ 新增：保存DQN动作选择历史记录 +++
        if USE_DQN and hasattr(agent, 'action_history'):
            history_filename = os.path.join(output_dir, f"{problem_name}_action_history.json")
            try:
                with open(history_filename, 'w') as f:
                    json.dump(agent.action_history, f, indent=4)
                print(f"DQN动作选择历史已保存到 '{history_filename}'")
            except Exception as e:
                print(f"[错误] 保存DQN动作历史时发生错误: {e}")
        # +++++++++++++++++++++++++++++++++++++


if __name__ == '__main__':
    
    # --- 批量处理模式 ---
    problem_instances_path = r"D:\BaiduSyncdisk\agricultural_robots\memetic_dqn_solver\memetic_dqn_solver\problem_instances"
    
    # 1. 发现所有算例文件
    all_instance_files = sorted(glob.glob(os.path.join(problem_instances_path, "*.json")))
    total_instances = len(all_instance_files)
    
    print(f"发现 {total_instances} 个算例，准备开始批量处理...")
    
    # 2. 循环求解
    for i, file_path in enumerate(all_instance_files):
        instance_name = os.path.basename(file_path)
        print(f"\n--- [{i+1}/{total_instances}] 正在处理: {instance_name} ---")
        
        # 调用封装好的求解函数
        solve_instance(file_path, show_plot=False)
        
    print(f"\n{'='*30} ALL DONE {'='*30}")
    print(f"全部 {total_instances} 个算例处理完毕。")
