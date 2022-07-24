

class DecisionTree:
    def __init__(self,
            attributes_map: dict,
            max_depth: int = 20,
            node_purity: float = 0.9):
        self._nodes = set()
        self._root_node = None
        self._max_depth = max_depth
        self._node_purity = node_purity
        # attributes map is a disctionary contianing the type of each attribute in the data.
        # Must be one of ['categorical', 'boolean', 'continuous']
        for attr_name in attributes_map.keys():
            if not attributes_map[attr_name] in ['continuous', 'categorical', 'boolean']:
                raise TypeError('Attribute type not supported')
        self._attributes_map = attributes_map

    def add_node(self, node:, parent_node) -> None:
        """ Add a node to the tree's set of nodes and connects it to its parent node """
        node.set_parent_node(parent_node)
        self._nodes.add(node)
        if not parent_node is None:
            parent_node.add_child(node)
        elif node.get_label() == 'root':
            self._root_node = node
        else:
            raise ValueError(f"Can't add node {node._label}. Parent label not present in the tree")

    def delete_node(self, node) -> None:
        """ Removes a node from the tree's set of nodes and disconnects it from its parent node """
        parent_node = node.get_parent_node()
        if node.get_parent_node() is None:
            raise ValueError(f"Can't delete node {node._label}. Parent node not found")
        parent_node.delete_child(node)
        self._nodes.remove(node)
