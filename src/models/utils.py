from torch_geometric.nn.pool import radius


def build_edges(
    query,
    database,
    indices=None,
    r_max=1.0,
    k_max=10,
    return_indices=False,
    self_loop=False,
    batch_index=None,
):
    edge_list = radius(
        database,
        query,
        r=r_max,
        max_num_neighbors=k_max,
        batch_x=batch_index,
        batch_y=batch_index,
    )

    # Reset indices subset to correct global index
    if indices is not None:
        edge_list[0] = indices[edge_list[0]]

    # Remove self-loops
    if not self_loop:
        edge_list = edge_list[:, edge_list[0] != edge_list[1]]

    return edge_list
