import random

def generate_subgraph(graph, num_start_nodes=None):
    # 创建一个字典来存储子图的节点和紧后节点集合
    subgraph = {}
    # 创建一个集合来存储已处理的节点
    visited = set()

    if num_start_nodes is None:
        # 如果没有指定起始节点数量，则默认选择一个起始节点
        num_start_nodes = 1
    # 选择任意多个起始节点
    start_nodes = random.choices(list(graph.keys()), k=num_start_nodes)
    # 将起始节点添加到子图中并标记为已处理
    for start_node in start_nodes:
        subgraph[start_node] = graph[start_node]
        visited.add(start_node)
    # 使用深度优先搜索(Depth-First Search, DFS)生成子图
    for start_node in start_nodes:
        dfs(graph, start_node, subgraph, visited)

    return subgraph


def dfs(graph, node, subgraph, visited):
    for successor in graph[node]:
        if successor not in visited:
            # 将节点的紧后节点添加到子图中并标记为已处理
            subgraph[successor] = graph[successor]
            visited.add(successor)

            # 递归调用DFS遍历紧后节点
            dfs(graph, successor, subgraph, visited)



if __name__ == '__main__':
    # 测试示例
    graph = {
        'A': ['B', 'C'],
        'B': ['C', 'D'],
        'C': ['D'],
        'D': []
    }

    subgraph = generate_subgraph(graph, num_start_nodes=random.randint(2, len(graph.keys())))
    print(subgraph)
