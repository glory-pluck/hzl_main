"""
如果用户只想更新图中的部分节点，
可以先通过想要囊括的节点编号创建一个子图， 
然后在子图上调用 update_all() 方法。例如：
这是小批量训练中的常见用法
"""
nid = [0, 2, 3, 6, 7, 9]
sg = g.subgraph(nid)
sg.update_all(message_func, reduce_func, apply_node_func)