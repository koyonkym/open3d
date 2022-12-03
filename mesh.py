import open3d as o3d
import open3d_example as o3dtut
import numpy as np
import copy

print('Testing mesh in Open3D...')
armadillo_mesh = o3d.data.ArmadilloMesh()
mesh = o3d.io.read_triangle_mesh(armadillo_mesh.path)

knot_mesh = o3d.data.KnotMesh()
mesh = o3d.io.read_triangle_mesh(knot_mesh.path)
print(mesh)
print('Vertices:')
print(np.asarray(mesh.vertices))
print('Triangles:')
print(np.asarray(mesh.triangles))

## Visualize a 3D mesh
print(
    'Try to render a mesh with normals (exist: ' +
    str(mesh.has_vertex_normals()) +
    ') and colors (exist: ' +
    str(mesh.has_vertex_colors()) +
    ')')
o3d.visualization.draw_geometries([mesh])
print('A mesh with no normals and no colors does not look good.')

## Surface normal estimation
print('Computing normal and rendering it.')
mesh.compute_vertex_normals()
print(np.asarray(mesh.triangle_normals))
o3d.visualization.draw_geometries([mesh])

## Crop mesh
print('We make a partial mesh of only the first half triangles.')
mesh1 = copy.deepcopy(mesh)
mesh1.triangles = o3d.utility.Vector3iVector(np.asarray(mesh1.triangles)[:len(mesh1.triangles) // 2, :])
mesh1.triangle_normals = o3d.utility.Vector3dVector(np.asarray(mesh1.triangle_normals)[:len(mesh1.triangle_normals) // 2, :])
print(mesh1.triangles)
o3d.visualization.draw_geometries([mesh1])

## Paint mesh
print('Painting the mesh')
mesh1.paint_uniform_color([1, 0.706, 0])
o3d.visualization.draw_geometries([mesh1])

## Mesh properties
def check_properties(name, mesh):
    mesh.compute_vertex_normals()

    edge_manifold = mesh.is_edge_manifold(allow_boundary_edges=True)
    edge_manifold_boundary = mesh.is_edge_manifold(allow_boundary_edges=False)
    vertex_manifold = mesh.is_vertex_manifold()
    self_intersecting = mesh.is_self_intersecting()
    watertight = mesh.is_watertight()
    orientable = mesh.is_orientable()

    print(name)
    print(f' edge_manifold:          {edge_manifold}')
    print(f' edge_manifold_boundary: {edge_manifold_boundary}')
    print(f' vertex_manifold:        {vertex_manifold}')
    print(f' self_intersecting:      {self_intersecting}')
    print(f' watertight:             {watertight}')
    print(f' orientable:             {orientable}')

    geoms = [mesh]
    if not edge_manifold:
        edges = mesh.get_non_manifold_edges(allow_boundary_edges=True)
        geoms.append(o3dtut.edges_to_lineset(mesh, edges, (1, 0, 0)))
    if not edge_manifold_boundary:
        edges = mesh.get_non_manifold_edges(allow_boundary_edges=False)
        geoms.append(o3dtut.edges_to_lineset(mesh, edges, (0, 1, 0)))
    if not vertex_manifold:
        verts = np.asarray(mesh.get_non_manifold_vertices())
        pcl = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(np.asarray(mesh.vertices)[verts]))
        pcl.paint_uniform_color((0, 0, 1))
        geoms.append(pcl)
    if self_intersecting:
        intersecting_triangles = np.asarray(mesh.get_self_intersecting_triangles())
        intersecting_triangles = intersecting_triangles[0:1]
        intersecting_triangles = np.unique(intersecting_triangles)
        print(' # visualize self-intersecting triangles')
        triangles = np.asarray(mesh.triangles)[intersecting_triangles]
        edges = [np.vstack((triangles[:, i], triangles[:, j])) for i, j in [(0, 1), (1, 2), (2, 0)]]
        edges = np.hstack(edges).T
        edges = o3d.utility.Vector2iVector(edges)
        geoms.append(o3dtut.edges_to_lineset(mesh, edges, (1, 0, 1)))
    o3d.visualization.draw_geometries(geoms, mesh_show_back_face=True)

knot_mesh_data = o3d.data.KnotMesh()
knot_mesh = o3d.io.read_triangle_mesh(knot_mesh_data.path)
check_properties('KnotMesh', knot_mesh)
check_properties('Mobius', o3d.geometry.TriangleMesh.create_mobius(twists=1))
check_properties('non-manifold edge', o3dtut.get_non_manifold_edge_mesh())
check_properties('non-manifold vertex', o3dtut.get_non_manifold_vertex_mesh())
check_properties('open box', o3dtut.get_open_box_mesh())
check_properties('intersecting_boxes', o3dtut.get_intersecting_boxes_mesh())
