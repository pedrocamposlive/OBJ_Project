import open3d as o3d
import numpy as np
import copy
import matplotlib.pyplot as plt # Adicione esta linha!

# --- FUNÇÕES AUXILIARES PARA GERAR PONTOS DA CANECA E DO PLANO ---
# (Copiadas dos scripts anteriores para que o script seja autocontido)

def create_cylinder_points(radius, height, num_points, z_offset=0.0, is_top=False, radius_inner_for_top=None, radius_outer_for_top=None):
    points = []
    for _ in range(num_points):
        theta = np.random.uniform(0, 2 * np.pi)
        h = np.random.uniform(0, height)
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        z = h + z_offset
        points.append([x, y, z])
    if is_top:
        if radius_inner_for_top is None or radius_outer_for_top is None:
            raise ValueError("radius_inner_for_top and radius_outer_for_top must be provided when is_top is True")
        for _ in range(num_points // 10):
            theta = np.random.uniform(0, 2 * np.pi)
            x = np.random.uniform(radius_inner_for_top, radius_outer_for_top) * np.cos(theta)
            y = np.random.uniform(radius_inner_for_top, radius_outer_for_top) * np.sin(theta)
            z = height + z_offset
            points.append([x, y, z])
    return np.array(points)

def create_disk_points(radius, num_points, z_position):
    points = []
    for _ in range(num_points):
        r = np.random.uniform(0, radius)
        theta = np.random.uniform(0, 2 * np.pi)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        points.append([x, y, z_position])
    return np.array(points)

def create_torus_segment_points(major_radius, minor_radius, num_points, start_angle, end_angle, center_offset_x, center_offset_y, center_offset_z):
    points = []
    for _ in range(num_points):
        u = np.random.uniform(start_angle, end_angle)
        v = np.random.uniform(0, 2 * np.pi)
        x = (major_radius + minor_radius * np.cos(v)) * np.cos(u) + center_offset_x
        y = (major_radius + minor_radius * np.cos(v)) * np.sin(u) + center_offset_y
        z = minor_radius * np.sin(v) + center_offset_z
        points.append([x, y, z])
    return np.array(points)

def generate_mug_pcd(outer_radius, inner_radius, height, num_points_body, num_points_bottom, num_points_handle):
    thickness_bottom = 0.005
    handle_radius = 0.02
    handle_tube_radius = 0.005
    
    pcd_outer_body = create_cylinder_points(outer_radius, height, num_points_body // 2, 
                                            is_top=True, radius_inner_for_top=inner_radius, 
                                            radius_outer_for_top=outer_radius)
    pcd_outer_body_top = create_disk_points(outer_radius, num_points_body // 10, height)

    pcd_inner_body = create_cylinder_points(inner_radius, height, num_points_body // 2)

    pcd_bottom_top = create_disk_points(outer_radius, num_points_bottom, thickness_bottom)
    pcd_bottom_bottom = create_disk_points(outer_radius, num_points_bottom, 0.0)

    handle_offset_y = -(outer_radius + handle_radius * 0.8)
    handle_offset_z = height / 2

    handle_rotation_angle = np.pi / 2
    rotation_matrix = np.array([
        [np.cos(handle_rotation_angle), -np.sin(handle_rotation_angle), 0],
        [np.sin(handle_rotation_angle),  np.cos(handle_rotation_angle), 0],
        [0, 0, 1]
    ])

    pcd_handle_top_arc = create_torus_segment_points(
        major_radius=handle_radius, minor_radius=handle_tube_radius,
        num_points=num_points_handle // 2,
        start_angle=np.pi * 0.05, end_angle=np.pi * 0.5,
        center_offset_x=0.0, center_offset_y=handle_offset_y, center_offset_z=handle_offset_z
    )

    pcd_handle_bottom_arc = create_torus_segment_points(
        major_radius=handle_radius, minor_radius=handle_tube_radius,
        num_points=num_points_handle // 2,
        start_angle=np.pi * 0.5, end_angle=np.pi * 0.95,
        center_offset_x=0.0, center_offset_y=handle_offset_y, center_offset_z=handle_offset_z
    )

    pcd_handle_top_arc_rotated = (rotation_matrix @ pcd_handle_top_arc.T).T
    pcd_handle_bottom_arc_rotated = (rotation_matrix @ pcd_handle_bottom_arc.T).T

    all_points = np.vstack([
        pcd_outer_body, pcd_inner_body, pcd_bottom_top, pcd_bottom_bottom,
        pcd_handle_top_arc_rotated, pcd_handle_bottom_arc_rotated
    ])

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_points)
    colors = np.tile(np.array([0.7, 0.4, 0.1]), (len(all_points), 1))
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd

def create_plane_points(width, length, num_points, z_position=0.0, color=[0.5, 0.5, 0.5]):
    """Gera pontos para um plano (simulando o chão)."""
    points = []
    for _ in range(num_points):
        x = np.random.uniform(-width / 2, width / 2)
        y = np.random.uniform(-length / 2, length / 2)
        points.append([x, y, z_position])
    pcd_plane = o3d.geometry.PointCloud()
    pcd_plane.points = o3d.utility.Vector3dVector(np.array(points))
    pcd_plane.paint_uniform_color(color)
    return pcd_plane

# --- FIM DAS FUNÇÕES AUXILIARES ---

# --- CONFIGURAÇÕES GERAIS ---
CAN_OUTER_RADIUS = 0.04
CAN_INNER_RADIUS = 0.038
CAN_HEIGHT = 0.1
CAN_NUM_POINTS_BODY = 10000
CAN_NUM_POINTS_BOTTOM = 2000
CAN_NUM_POINTS_HANDLE = 4000

PLANE_WIDTH = 1.0  # 1 metro de largura
PLANE_LENGTH = 1.0 # 1 metro de comprimento
PLANE_NUM_POINTS = 50000 # Muitos pontos para o chão

# Parâmetros para segmentação de plano (RANSAC)
distance_threshold = 0.01  # Distância máxima de um ponto ao plano para ser considerado inlier (1 cm)
ransac_n = 3               # Número de pontos para amostrar aleatoriamente para estimar um plano
num_iterations = 1000      # Número de iterações do RANSAC

# Parâmetros para clustering (DBScan)
eps = 0.05                 # Raio de busca para vizinhos (5 cm)
min_points = 10            # Número mínimo de pontos para formar um cluster

# --- INÍCIO DO SCRIPT ---
print("--- Iniciando o Teste de Segmentação Básica ---")

# 1. Gerar a nuvem de pontos do ambiente (chão) e do objeto (caneca)
print("\n1. Gerando a nuvem de pontos do CHÃO (cinza) e da CANECA (laranja)...")

# Gerar o plano (chão)
pcd_plane = create_plane_points(PLANE_WIDTH, PLANE_LENGTH, PLANE_NUM_POINTS, z_position=0.0)

# Gerar a caneca e posicioná-la ligeiramente acima do chão
pcd_mug = generate_mug_pcd(CAN_OUTER_RADIUS, CAN_INNER_RADIUS, CAN_HEIGHT,
                           CAN_NUM_POINTS_BODY, CAN_NUM_POINTS_BOTTOM, CAN_NUM_POINTS_HANDLE)
# Mover a caneca para uma posição específica (acima do chão e um pouco para o lado)
mug_translation = np.array([0.1, 0.1, 0.005]) # 0.5 cm acima do chão
pcd_mug.translate(mug_translation)


# Combinar a caneca e o plano em uma única nuvem de pontos "bruta"
pcd_combined_raw = pcd_plane + pcd_mug
print(f"   Nuvem de pontos combinada (Chão + Caneca): {len(pcd_combined_raw.points)} pontos.")

# 2. Visualizar a Nuvem de Pontos Combinada Original
print("\n2. Visualizando a nuvem combinada ORIGINAL. Feche a janela para continuar...")
o3d.visualization.draw_geometries([pcd_combined_raw],
                                  window_name="Nuvem Combinada Original",
                                  width=800, height=600, left=50, top=50)
print("   Visualização da nuvem original encerrada.")


# 3. Remover o Plano (Chão) usando RANSAC
print(f"\n3. Removendo o plano (chão) usando RANSAC com:")
print(f"   - Limiar de distância: {distance_threshold} m")
print(f"   - Iterações: {num_iterations}")

# segment_plane retorna:
#   - model: Coeficientes do plano (ax + by + cz + d = 0)
#   - inliers: Índices dos pontos que pertencem ao plano
#   - outliers: Índices dos pontos que não pertencem ao plano (seu objeto!)
model, inliers = pcd_combined_raw.segment_plane(distance_threshold=distance_threshold,
                                               ransac_n=ransac_n,
                                               num_iterations=num_iterations)

# Separar os pontos do plano (inliers) dos pontos do objeto (outliers)
pcd_plane_removed = pcd_combined_raw.select_by_index(inliers, invert=True) # Pontos que NÃO são do plano
pcd_plane_itself = pcd_combined_raw.select_by_index(inliers) # Pontos que SÃO do plano

print(f"   Pontos do plano removidos. Nuvem restante: {len(pcd_plane_removed.points)} pontos.")

# Visualizar a nuvem após a remoção do plano (deve ser só a caneca e talvez algum ruído)
print("\n4. Visualizando a nuvem APÓS REMOÇÃO DO PLANO. Feche a janela para continuar...")
o3d.visualization.draw_geometries([pcd_plane_removed],
                                  window_name="Nuvem Sem Plano",
                                  width=800, height=600, left=900, top=50)
print("   Visualização da nuvem sem plano encerrada.")


# 5. Agrupamento (Clustering) para Isolar Objetos (a Caneca)
print(f"\n5. Aplicando Clustering (DBScan) para isolar objetos com:")
print(f"   - Raio (eps): {eps} m")
print(f"   - Mínimo de pontos por cluster: {min_points}")

# dbscan retorna os rótulos de cluster para cada ponto
# -1 significa ruído (pontos que não pertencem a nenhum cluster)
labels = np.array(pcd_plane_removed.cluster_dbscan(eps=eps, min_points=min_points, print_progress=True))

max_label = labels.max()
print(f"   Número total de clusters detectados: {max_label + 1} (incluindo ruído -1).")

# Visualizar os clusters com cores diferentes
colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
colors[labels < 0] = 0 # Pontos de ruído (não clusterizados) em preto
pcd_plane_removed.colors = o3d.utility.Vector3dVector(colors[:, :3])

print("\n6. Visualizando os CLUSTERS DETECTADOS. Feche a janela para continuar...")
o3d.visualization.draw_geometries([pcd_plane_removed],
                                  window_name="Clusters Detectados",
                                  width=800, height=600, left=50, top=400)
print("   Visualização dos clusters encerrada.")

# 7. Separar o Maior Cluster (que deve ser a caneca)
# Encontre o cluster com mais pontos (geralmente o objeto principal)
if max_label >= 0:
    unique_labels, counts = np.unique(labels, return_counts=True)
    # Ignorar o cluster de ruído (-1)
    valid_labels_idx = unique_labels[unique_labels >= 0]
    valid_counts = counts[unique_labels >= 0]

    if len(valid_labels_idx) > 0:
        main_cluster_label = valid_labels_idx[np.argmax(valid_counts)]
        print(f"   Identificado o maior cluster (objeto principal) com o label: {main_cluster_label}")
        pcd_object = pcd_plane_removed.select_by_index(np.where(labels == main_cluster_label)[0])
        pcd_object.paint_uniform_color([0.8, 0.2, 0.8]) # Pinta o objeto de destaque de roxo

        print("\n7. Visualizando o OBJETO ISOLADO. Feche a janela para continuar...")
        o3d.visualization.draw_geometries([pcd_object],
                                          window_name="Objeto Isolado",
                                          width=800, height=600, left=900, top=400)
        print("   Visualização do objeto isolado encerrada.")

        # Salvar o objeto isolado
        output_object_file = "objeto_isolado_caneca.ply"
        print(f"\n   Salvando o objeto isolado em: {output_object_file}")
        o3d.io.write_point_cloud(output_object_file, pcd_object)
    else:
        print("   Nenhum cluster válido (com pontos suficientes) foi detectado além do ruído.")
else:
    print("   Nenhum cluster foi detectado.")

print("\n--- Teste de Segmentação Básica Concluído! ---")
