import numpy as np
import os
import ngsolve as ng
from new_pignn.graph_creator import GraphCreator
from new_pignn.mesh_utils import (
    create_rectangular_mesh,
    create_gaussian_initial_condition,
    create_free_node_subgraph,
)


def main():
    mesh = create_rectangular_mesh(width=1.0, height=0.6, maxh=0.3)
    n_points = len(list(mesh.ngmesh.Points()))
    print(f"   Created mesh with {n_points} nodes")

    dirichlet_names = ["left", "right", "top", "bottom"]
    neumann_names = []  # No Neumann boundaries for this example

    # First create a dummy graph to get positions
    temp_creator = GraphCreator(
        mesh,
        n_neighbors=2,
        dirichlet_names=dirichlet_names,
        neumann_names=neumann_names,
        connectivity_method="fem",
    )
    temp_data, _ = temp_creator.create_graph()

    # Create Gaussian initial condition
    T_initial = create_gaussian_initial_condition(
        temp_data.pos,
        num_gaussians=2,
        amplitude_range=(0.5, 1.0),
        sigma_fraction_range=(0.1, 0.2),
        seed=42,
        centered=True,
        enforce_boundary_conditions=True,
    )
    print(f"   Temperature range: [{T_initial.min():.3f}, {T_initial.max():.3f}]")

    connectivity_methods = ["fem", "knn"]

    for method in connectivity_methods:

        # Create GraphCreator
        if method == "knn":
            creator = GraphCreator(
                mesh,
                n_neighbors=2,
                dirichlet_names=dirichlet_names,
                neumann_names=neumann_names,
                connectivity_method="knn",
            )
        else:
            creator = GraphCreator(
                mesh,
                n_neighbors=2,
                dirichlet_names=dirichlet_names,
                neumann_names=neumann_names,
                connectivity_method="fem",
            )

        # Create graph
        graph_data, aux_data = creator.create_graph(T_current=T_initial, t_scalar=0.0)

        # Print statistics
        creator.print_graph_stats(graph_data, aux_data)

        # Visualize graph
        print(f"Visualizing {method.upper()} graph...")
        creator.visualize_graph(
            graph_data,
            aux_data,
            figsize=(14, 6),
            node_size=30,
            save_path=f"graph_visualization_{method}.png",
        )

    creator = GraphCreator(
        mesh,
        dirichlet_names=dirichlet_names,
        neumann_names=neumann_names,
        connectivity_method="fem",
    )
    full_graph, aux = creator.create_graph(T_current=T_initial, t_scalar=0.0)

    # Create subgraph with only free (non-Dirichlet) nodes
    free_graph, node_mapping = create_free_node_subgraph(full_graph, aux)
    # Visualize free node subgraph
    creator.visualize_graph(
        free_graph,
        aux,
        figsize=(14, 6),
        node_size=30,
        save_path="graph_visualization_free_nodes.png",
        only_free_nodes=True,
    )

    print("Demo completed! Check the generated visualization files:")
    print("  - graph_visualization_fem.png")
    print("  - graph_visualization_knn.png")
    print("  - graph_visualization_free_nodes.png")


if __name__ == "__main__":
    main()
