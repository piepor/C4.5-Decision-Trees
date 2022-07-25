from attributes import AttributeType

def check_attributes(attributes, parent_node):
    """ checks if the attributes are correct """
    if attributes.node_name == "root" and not attributes.node_level == 0:
        raise ValueError("Root node must have level 0")
    if attributes.node_level < 0:
        raise ValueError(f"Level must be positive. \
                Node [{attributes.node_name}] - Parent [{parent_node.get_label()}")
    if attributes.attribute_type == AttributeType.CONTINUOUS and not attributes.threshold:
        parent_name = None
        if parent_node:
            parent_name = parent_node.get_label() # the root node does not have a parent node
        raise ValueError(f"A continuous type must have a threshold. \
                Node [{attributes.node_name}] - Parent [{parent_name}")
    if not attributes.attribute_type == AttributeType.CONTINUOUS and attributes.threshold:
        parent_name = None
        if parent_node:
            parent_name = parent_node.get_label() # the root node does not have a parent node
        raise ValueError(f"A threshold has been set on non continuous node type. \
                Node [{attributes.node_name}] - Parent [{parent_name}")
