from c4dot5.nodes import LeafNode


def create_label_leaf_node(node: LeafNode):
    label = f"{node.get_label()} \n [Classes]:"
    classes = node.get_classes()
    for target, num_instances in classes.items():
        label = f"{label} \n - {target}: {num_instances}"
    return label
        
