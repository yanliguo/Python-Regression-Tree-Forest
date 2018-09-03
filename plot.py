from io import BytesIO


def _plot_node(graph, root, emurate_idx, 
        yes_color, no_color, leaf_shape, node_shape, 
        feature_idx_to_name=None):
    """
    recursively plot nodes, 
    root: Tree
    feature_idx_to_name: dict[int -> str]
    returns parsed node name
    """
    if root is None:
        return None, emurate_idx
    emurate_idx = emurate_idx + 1
    # node 
    if root.is_tree():
        name = 'node' + str(emurate_idx)
        if feature_idx_to_name is None or root.split_var is None or \
                root.split_var not in feature_idx_to_name:
            feature = 'f' + str(root.split_var)
        else:
            feature = feature_idx_to_name.get(root.split_var)
        label = feature + '<' + str(root.split_val)
        graph.node(name, label, shape=node_shape)
        left_name, left_emurate_idx = _plot_node(graph, root.left, emurate_idx, 
                yes_color, no_color, leaf_shape, node_shape, feature_idx_to_name)
        if left_name is not None:
            graph.edge(name, left_name, label='yes', color=yes_color)
        right_name, right_emurate_idx = _plot_node(graph, root.right, left_emurate_idx, \
                yes_color, no_color, leaf_shape, node_shape, feature_idx_to_name)
        if right_name is not None:
            graph.edge(name, right_name, label='no', color=no_color)
        return name, right_emurate_idx
    else:
        name = 'leaf' + str(emurate_idx)
        label = 'leaf=' + str(root.predict)
        graph.node(name, label=label, shape=leaf_shape)
    return name, emurate_idx



def _to_graphviz(cart, rankdir='UT', yes_color='#0000FF', no_color='#FF0000',
        leaf_shape='box', node_shape='circle', **kwargs):
    try:
        from graphviz import Digraph
    except ImportError:
        raise ImportError('Missing graphviz')

    kwargs = kwargs.copy()
    kwargs.update({'rankdir': rankdir})
    graph = Digraph(graph_attr=kwargs)

    _plot_node(graph, cart, 0, yes_color, no_color, leaf_shape, node_shape, None)

    return graph


def plot_cart(cart, _format='png', rankdir='UT', ax=None, **kwargs):
    try:
        import matplotlib.pyplot as plt
        import matplotlib.image as image
        from PIL import Image
    except ImportError:
        raise ImportError('Missing lib matplotlib')

    if ax is None:
        _, ax = plt.subplots(1, 1)

    graph = _to_graphviz(cart, rankdir=rankdir, **kwargs)

    bytesIO = BytesIO()
    bytesIO.write(graph.pipe(format=_format))
    bytesIO.seek(0)
    # display using PIL instead
    with Image.open(bytesIO) as img:
        img.show()
    # img = image.imread(bytesIO)
    # ax.imshow(img)
    # ax.axis('off')
    return ax


def plot_feature_importance(feature_wts, filename=None, title='Feature weights'):
    try:
        import matplotlib.pyplot as plt
        from PIL import Image
    except:
        raise ImportError('matplot missing')

    names = list(feature_wts.keys())
    values = list(map(lambda x: feature_wts[x], names))
    plt.barh(range(len(values)), values, tick_label=names)
    plt.title(title)
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img = Image.open(buf)
    img.show()
    buf.close()




