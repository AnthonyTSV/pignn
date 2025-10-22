import numpy as np
import os
import ngsolve as ng
from new_pignn.graph_creator import GraphCreator
from new_pignn.mesh_utils import (
    create_rectangular_mesh,
    create_gaussian_initial_condition,
    create_neumann_values,
    create_dirichlet_values,
)


def main():
    mesh = create_rectangular_mesh(width=1.0, height=0.6, maxh=0.2)
    n_points = len(list(mesh.ngmesh.Points()))
    print(f"   Created mesh with {n_points} nodes")

    dirichlet_names = ["left", "right"]
    neumann_names = ["top", "bottom"]

    # First create a dummy graph to get positions and auxiliary data
    temp_creator = GraphCreator(
        mesh,
        n_neighbors=2,
        dirichlet_names=dirichlet_names,
        neumann_names=neumann_names,
        connectivity_method="fem",
    )
    temp_data, temp_aux = temp_creator.create_graph()

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
    
    # Create Neumann values with boundary-specific constant values
    neumann_boundary_values = {"top": 1, "bottom": 0}  # Constant flux on top boundary
    neumann_vals = temp_creator.create_neumann_values(
        temp_data.pos,
        temp_aux,
        neumann_names,
        flux_values=neumann_boundary_values,
        seed=42
    )
    print(f"   Neumann values range: [{neumann_vals.min():.3f}, {neumann_vals.max():.3f}]")
    neumann_mask = temp_aux['neumann_mask']
    if neumann_mask.any():
        neumann_nodes = neumann_vals[neumann_mask.numpy()]
        print(f"   Neumann nodes flux range: [{neumann_nodes.min():.3f}, {neumann_nodes.max():.3f}]")

    # Create Dirichlet values with boundary-specific constant values
    dirichlet_boundary_values = {
        "left": 5,
        "right": 5,
    }
    dirichlet_vals = temp_creator.create_dirichlet_values(
        temp_data.pos,
        temp_aux,
        dirichlet_names,
        boundary_values=dirichlet_boundary_values,
        seed=42
    )
    print(f"   Dirichlet values range: [{dirichlet_vals.min():.3f}, {dirichlet_vals.max():.3f}]")
    dirichlet_mask = temp_aux['dirichlet_mask']
    if dirichlet_mask.any():
        dirichlet_nodes = dirichlet_vals[dirichlet_mask.numpy()]
        print(f"   Dirichlet nodes range: [{dirichlet_nodes.min():.3f}, {dirichlet_nodes.max():.3f}]")

    creator = GraphCreator(
        mesh,
        n_neighbors=2,
        dirichlet_names=dirichlet_names,
        neumann_names=neumann_names,
        connectivity_method="fem",
    )

    # Create graph with Neumann and Dirichlet values
    graph_data, aux_data = creator.create_graph(T_current=T_initial, t_scalar=0.0, neumann_values=neumann_vals, dirichlet_values=dirichlet_vals)

    # Print statistics
    creator.print_graph_stats(graph_data, aux_data)

    # Visualize graph
    creator.visualize_graph(
        graph_data,
        aux_data,
        figsize=(20, 5),
        node_size=30,
        save_path="graph_visualization_fem.png",
    )

    creator = GraphCreator(
        mesh,
        dirichlet_names=dirichlet_names,
        neumann_names=neumann_names,
        connectivity_method="fem",
    )
    full_graph, aux = creator.create_graph(T_current=T_initial, t_scalar=0.0, neumann_values=neumann_vals, dirichlet_values=dirichlet_vals)

    # Create subgraph with only free (non-Dirichlet) nodes
    free_graph, node_mapping, new_aux = creator.create_free_node_subgraph(full_graph, aux)
    # Visualize free node subgraph
    creator.visualize_graph(
        free_graph,
        new_aux,
        figsize=(20, 5),
        node_size=30,
        save_path="graph_visualization_free_nodes.png",
    )

    print("Demo completed! Check the generated visualization files:")
    print("  - graph_visualization_fem.png (with Dirichlet and Neumann values)")
    print("  - graph_visualization_free_nodes.png (free nodes subgraph with Dirichlet values)")


if __name__ == "__main__":
    main()
