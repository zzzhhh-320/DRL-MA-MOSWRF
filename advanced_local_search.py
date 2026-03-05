import random
import numpy as np
import copy

# =================================================================================
# 辅助函数
# =================================================================================

def _calculate_route_finish_time(route, robot_id, params):
    """计算单条路径的完成时间"""
    p = params
    t_cur = 0.0
    path = [0] + route + [0]
    for i in range(len(path) - 1):
        a, b = path[i], path[i+1]
        travel = p['dist'][(a, b)] / p['v_base'][robot_id-1]
        arrive = t_cur + travel
        if b != 0:
            s_start = max(arrive, p['TW'][b][0])
            t_cur = s_start + p['tau'][b]
        else:
            t_cur = arrive
    return t_cur

def _calculate_total_distance(routes, params):
    """计算一个完整解的总距离"""
    total_dist = 0
    for r, tasks in routes.items():
        path = [0] + tasks + [0]
        for i in range(len(path) - 1):
            total_dist += params['dist'][(path[i], path[i+1])]
    return total_dist

def _find_best_insertion(task, routes, params, exclude_robot_id=None):
    """为单个任务在所有路径中寻找最佳插入位置（最小距离增量）"""
    best_robot, best_pos, min_delta = -1, -1, float('inf')
    
    for r_id, path in routes.items():
        if r_id == exclude_robot_id:
            continue
        
        # 检查容量
        current_load = sum(params['q'][t] for t in path)
        if current_load + params['q'][task] > params['Q'][r_id-1]:
            continue

        # 遍历所有可能的插入位置
        for i in range(len(path) + 1):
            prev_node = path[i-1] if i > 0 else 0
            next_node = path[i] if i < len(path) else 0
            
            delta = params['dist'][(prev_node, task)] + params['dist'][(task, next_node)] - params['dist'][(prev_node, next_node)]
            
            if delta < min_delta:
                min_delta = delta
                best_robot = r_id
                best_pos = i
                
    return best_robot, best_pos, min_delta

# =================================================================================
# 四大高级局部搜索算子
# =================================================================================

def critical_task_push(routes, params):
    """
    1. 关键任务推送 (Critical Task Push, CTP)
    解决负载不均衡问题。
    """
    new_routes = {r: list(p) for r, p in routes.items()}

    # 1. 识别受害者和关键任务
    finish_times = {r: _calculate_route_finish_time(p, r, params) for r, p in new_routes.items()}
    victim_robot = max(finish_times, key=finish_times.get)
    
    if not new_routes[victim_robot]:
        return routes, [] # 受害者路径为空，无法操作

    key_task = new_routes[victim_robot][-1]

    # 2. 寻找最佳接收者和插入位置
    best_receiver, best_pos, _ = _find_best_insertion(key_task, new_routes, params, exclude_robot_id=victim_robot)

    # 3. 执行转移
    if best_receiver != -1:
        new_routes[victim_robot].pop()
        new_routes[best_receiver].insert(best_pos, key_task)
        return new_routes, [victim_robot, best_receiver]

    return routes, [] # 未找到可行的转移，返回原解

def coordinated_cluster_transfer(routes, params):
    """
    2. 协同集群转移 (Coordinated Cluster Transfer, CCT)
    优化路径的局部连贯性。
    """
    new_routes = {r: list(p) for r, p in routes.items()}

    # 1. 识别受害者
    finish_times = {r: _calculate_route_finish_time(p, r, params) for r, p in new_routes.items()}
    victim_robot = max(finish_times, key=finish_times.get)
    
    victim_path = new_routes[victim_robot]
    if len(victim_path) < 2:
        return routes, [] # 路径太短，无法形成簇

    # 2. 识别任务簇 (随机选择2-3个连续任务)
    cluster_size = random.randint(2, min(3, len(victim_path)))
    start_index = random.randint(0, len(victim_path) - cluster_size)
    cluster = victim_path[start_index : start_index + cluster_size]
    
    # 3. 寻找最佳接收者
    cluster_demand = sum(params['q'][t] for t in cluster)
    best_receiver, best_pos, min_delta = -1, -1, float('inf')

    for r_id, path in new_routes.items():
        if r_id == victim_robot:
            continue
        
        current_load = sum(params['q'][t] for t in path)
        if current_load + cluster_demand > params['Q'][r_id-1]:
            continue

        # 4. 确定最佳插入位置
        for i in range(len(path) + 1):
            prev_node = path[i-1] if i > 0 else 0
            next_node = path[i] if i < len(path) else 0
            
            # 计算距离增量
            dist_added = params['dist'][(prev_node, cluster[0])] + params['dist'][(cluster[-1], next_node)]
            internal_cluster_dist = sum(params['dist'][(cluster[j], cluster[j+1])] for j in range(len(cluster)-1))
            dist_removed = params['dist'][(prev_node, next_node)]
            
            delta = dist_added + internal_cluster_dist - dist_removed
            if delta < min_delta:
                min_delta = delta
                best_receiver = r_id
                best_pos = i

    # 5. 执行转移
    if best_receiver != -1:
        del new_routes[victim_robot][start_index : start_index + cluster_size]
        new_routes[best_receiver] = new_routes[best_receiver][:best_pos] + cluster + new_routes[best_receiver][best_pos:]
        return new_routes, [victim_robot, best_receiver]

    return routes, []

def route_ejection_greedy_absorption(routes, params):
    """
    3. 路径重构与吸收 (Route Ejection and Greedy Absorption, REGA)
    对一个"糟糕"的路径进行彻底重构，以摆脱局部最优。
    """
    new_routes = {r: list(p) for r, p in routes.items()}
    
    # 1. 识别受害者
    finish_times = {r: _calculate_route_finish_time(p, r, params) for r, p in new_routes.items()}
    victim_robot = max(finish_times, key=finish_times.get)

    if not new_routes[victim_robot]:
        return routes, []

    # 2. 弹出任务
    tasks_to_reinsert = new_routes[victim_robot]
    new_routes[victim_robot] = []
    random.shuffle(tasks_to_reinsert)

    # 3. 贪婪吸收
    changed_robots = set()
    for task in tasks_to_reinsert:
        best_robot, best_pos, _ = _find_best_insertion(task, new_routes, params)
        if best_robot != -1:
            new_routes[best_robot].insert(best_pos, task)
            changed_robots.add(best_robot)
        else:
            # 如果找不到任何合法位置（极不可能，除非容量设置有问题），则放回原处
            new_routes[victim_robot].append(task)
            changed_robots.add(victim_robot)
            
    return new_routes, list(changed_robots)

def precedence_aware_adaptive_reordering(routes, params):
    """
    4. 任务依赖感知的自适应重排 (Precedence-Aware Adaptive Reordering, PAAR)
    注意：这是一个修复与优化的算子，但在DQN框架中，我们假设输入解是可行的。
    因此，这里实现为一个基于约束的局部优化，而不是一个修复算子。
    它通过重新排序一小段路径来寻求更优的解。
    """
    new_routes = {r: list(p) for r, p in routes.items()}
    
    # 随机选择一个机器人进行优化
    robot_id = random.choice(list(new_routes.keys()))
    path = new_routes[robot_id]
    
    if len(path) < 3:
        return routes, [] # 路径太短，无法重排

    # 随机选择一个子路径进行重排
    sub_path_size = random.randint(3, min(len(path), 10))
    start_index = random.randint(0, len(path) - sub_path_size)
    sub_path = path[start_index : start_index + sub_path_size]

    # 构建子路径内的先后约束
    predecessors = {t: set() for t in sub_path}
    for u, v in params['P']:
        if u in sub_path and v in sub_path:
            predecessors[v].add(u)

    # 贪婪地、遵守约束地重建子路径
    reordered_sub_path = []
    completed_in_sub = set()
    
    while len(reordered_sub_path) < len(sub_path):
        candidates = [t for t in sub_path if t not in reordered_sub_path and predecessors[t].issubset(completed_in_sub)]
        
        if not candidates:
            # 如果出现循环依赖或问题，则无法继续，返回原解
            return routes, [] 

        # 从当前位置选择最近的候选任务
        current_pos = reordered_sub_path[-1] if reordered_sub_path else (path[start_index-1] if start_index > 0 else 0)
        best_next_task = min(candidates, key=lambda t: params['dist'][(current_pos, t)])
        
        reordered_sub_path.append(best_next_task)
        completed_in_sub.add(best_next_task)

    # 用重排后的子路径替换原始子路径
    new_routes[robot_id] = path[:start_index] + reordered_sub_path + path[start_index + sub_path_size:]

    return new_routes, [robot_id]

def multi_route_synergy_refinement(routes, params):
    """
    旨在通过交换不同机器人路径上的任务来减少总距离。
    """
    new_routes = {r: list(t) for r, t in routes.items()}
    
    # 筛选出有任务的路径
    active_routes_ids = [r for r, path in new_routes.items() if path]
    if len(active_routes_ids) < 2:
        return routes, [] # 至少需要两条路径才能交换

    # 随机选择两条不同的路径
    r_a_id, r_b_id = random.sample(active_routes_ids, 2)
    route_a, route_b = new_routes[r_a_id], new_routes[r_b_id]

    best_t_a_idx, best_t_b_idx, max_dist_reduction = -1, -1, -float('inf')

    # 遍历两条路径的所有任务对
    for i, t_a in enumerate(route_a):
        for j, t_b in enumerate(route_b):
            # 检查交换后的容量可行性
            load_a = sum(params['q'][t] for t in route_a)
            load_b = sum(params['q'][t] for t in route_b)
            q_a, q_b = params['q'][t_a], params['q'][t_b]
            
            if load_a - q_a + q_b > params['Q'][r_a_id-1] or \
               load_b - q_b + q_a > params['Q'][r_b_id-1]:
                continue

            # 计算交换带来的距离变化
            # 对路径A
            prev_a = route_a[i-1] if i > 0 else 0
            next_a = route_a[i+1] if i < len(route_a) - 1 else 0
            dist_change_a = (params['dist'][(prev_a, t_b)] + params['dist'][(t_b, next_a)]) - \
                            (params['dist'][(prev_a, t_a)] + params['dist'][(t_a, next_a)])
            
            # 对路径B
            prev_b = route_b[j-1] if j > 0 else 0
            next_b = route_b[j+1] if j < len(route_b) - 1 else 0
            dist_change_b = (params['dist'][(prev_b, t_a)] + params['dist'][(t_a, next_b)]) - \
                            (params['dist'][(prev_b, t_b)] + params['dist'][(t_b, next_b)])

            total_reduction = -(dist_change_a + dist_change_b) # 我们希望减少量最大化
            if total_reduction > max_dist_reduction:
                max_dist_reduction = total_reduction
                best_t_a_idx, best_t_b_idx = i, j

    # 如果找到了一个能减少总距离的交换，则执行交换
    if max_dist_reduction > 0:
        t_a_val = route_a[best_t_a_idx]
        route_a[best_t_a_idx] = route_b[best_t_b_idx]
        route_b[best_t_b_idx] = t_a_val
        return new_routes, [r_a_id, r_b_id]
        
    return new_routes, []


def perturbation_local_search(routes, params, perturbation_level=0.3):
    
    new_routes = copy.deepcopy(routes)
    all_tasks = [task for route in routes.values() for task in route]
    
    if not all_tasks:
        return routes, []

    num_to_perturb = int(len(all_tasks) * perturbation_level)
    if num_to_perturb == 0:
        return routes, []

    tasks_to_perturb = random.sample(all_tasks, num_to_perturb)
    
    # 1. 从路径中移除任务
    for r_id, route in new_routes.items():
        new_routes[r_id] = [t for t in route if t not in tasks_to_perturb]

    # 2. 高度随机化的重插入
    for task in tasks_to_perturb:
        # 随机选择一个机器人
        r_id = random.randint(1, params['K'])
        
        # 随机选择一个插入位置
        route = new_routes.get(r_id, [])
        pos = random.randint(0, len(route))
        
        new_routes[r_id].insert(pos, task)

    return new_routes, list(range(1, params['K'] + 1))


def large_scale_destruction_reconstruction(routes, params):
    
    new_routes = copy.deepcopy(routes)

    # 1. 破坏 (Destroy)
    destruction_rate = random.uniform(0.15, 0.35) # 随机破坏率，增加多样性
    all_tasks = [task for route in new_routes.values() for task in route]
    
    if not all_tasks:
        return routes, []

    num_to_destroy = int(len(all_tasks) * destruction_rate)
    if num_to_destroy == 0:
        # 确保即使比例很小，至少也破坏一个任务
        num_to_destroy = 1 if all_tasks else 0
    
    if num_to_destroy == 0:
        return routes, []

    tasks_to_reinsert = random.sample(all_tasks, num_to_destroy)
    
    # 从路径中移除任务
    for r_id in new_routes:
        new_routes[r_id] = [t for t in new_routes[r_id] if t not in tasks_to_reinsert]

    # 2. 重构 (Recreate)
    # 随机化重构顺序
    random.shuffle(tasks_to_reinsert)
    
    unassigned_tasks = []

    # 阶段一: 贪婪插入（遵守容量约束）
    for task in tasks_to_reinsert:
        best_robot, best_pos, _ = _find_best_insertion(task, new_routes, params)
        if best_robot != -1:
            new_routes[best_robot].insert(best_pos, task)
        else:
            unassigned_tasks.append(task)
    
    # 阶段二: 强制插入（处理阶段一无法插入的任务）
    if unassigned_tasks:
        for task in unassigned_tasks:
            # 找到导致最小容量超载的插入位置
            best_robot, best_pos, min_overload = -1, -1, float('inf')
            
            for r_id, path in new_routes.items():
                current_load = sum(params['q'][t] for t in path)
                overload = (current_load + params['q'][task]) - params['Q'][r_id-1]
                
                if overload < min_overload:
                    min_overload = overload
                    best_robot = r_id

            # 在找到的最佳（或最不差）的机器人中随机找个位置插入
            if best_robot != -1:
                pos = random.randint(0, len(new_routes[best_robot]))
                new_routes[best_robot].insert(pos, task)
            else:
                # 理论上不应该发生，除非没有机器人
                # 最后的备用策略：随便找个机器人插入
                fallback_robot = random.randint(1, params['K'])
                pos = random.randint(0, len(new_routes[fallback_robot]))
                new_routes[fallback_robot].insert(pos, task)

    return new_routes, list(range(1, params['K'] + 1))