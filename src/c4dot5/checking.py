from c4dot5.attributes import AttributeType

def check_attributes(node_attributes, parent_node):
    """ checks if the node attributes are correct """
    if node_attributes.node_name == "root" and not node_attributes.node_level == 0:
        raise ValueError("Root node must have level 0")
    if node_attributes.node_level < 0:
        raise ValueError(f"Level must be positive. \
                Node [{node_attributes.node_name}] - Parent [{parent_node.get_label()}")
    if node_attributes.attribute_type == AttributeType.CONTINUOUS and not node_attributes.threshold:
        parent_name = None
        if parent_node:
            parent_name = parent_node.get_label() # the root node does not have a parent node
        raise ValueError(f"A continuous type must have a threshold. \
                Node [{node_attributes.node_name}] - Parent [{parent_name}")
    if not node_attributes.attribute_type == AttributeType.CONTINUOUS and node_attributes.threshold:
        parent_name = None
        if parent_node:
            parent_name = parent_node.get_label() # the root node does not have a parent node
        raise ValueError(f"A threshold has been set on non continuous node type. \
                Node [{node_attributes.node_name}] - Parent [{parent_name}]")
