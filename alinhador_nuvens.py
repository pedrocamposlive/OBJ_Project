import open3d as o3d
import numpy as np
import copy # Para duplicar objetos sem referência

# --- FUNÇÕES AUXILIARES PARA GERAR PONTOS DA CANECA (COPIADAS DO SCRIPT ANTERIOR) ---
# Essas funções foram incluídas aqui para que o script seja autocontido.

# CORREÇÃO: Adicione radius_inner_for_top e radius_outer_for_top como parâmetros
def create_cylinder_points(radius, height, num_points, z_offset=0.0, is_top=False, radius_inner_for_top=None, radius_outer_for_top=None):
    """Gera pontos para a superfície de um cilindro."""
    points = []
    for _ in range(num_points):
        theta = np.random.uniform(0, 2 * np.pi)
        h = np.random.uniform(0, height)
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        z = h + z_offset
        points.append([x, y, z])
    if is_top: # Adiciona pontos para a parte superior (borda) se for um cilindro oco
        # CORREÇÃO AQUI: Usar os parâmetros passados
        if radius_inner_for_top is None or radius_outer_for_top is None:
            raise ValueError("radius_inner_for_top and radius_outer_for_top must be provided when is_top is True")
        for _ in range(num_points // 10): # Menos pontos para a borda
            theta = np.random.uniform(0, 2 * np.pi)
            x = np.random.uniform(radius_inner_for_top, radius_outer_for_top) * np.cos(theta)
            y = np.random.uniform(radius_inner_for_top, radius_outer_for_top) * np.sin(theta)
            z = height + z_offset
            points.append([x, y, z])
    return np.array(points)

def create_disk_points(radius, num_points, z_position):
    """Gera pontos para um disco plano."""
    points = []
    for _ in range(num_points):
        r = np.random.uniform(0, radius)
        theta = np.random.uniform(0, 2 * np.pi)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        points.append([x, y, z_position])
    return np.array(points)

def create_torus_segment_points(major_radius, minor_radius, num_points, start_angle, end_angle, center_offset_x, center_offset_y, center_offset_z):
    """Gera pontos para um segmento de toro (para a alça)."""
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
    """Gera uma nuvem de pontos de uma caneca."""
    thickness_bottom = 0.005
    handle_radius = 0.02
    handle_tube_radius = 0.005
    
    # CORREÇÃO AQUI: Passar inner_radius e outer_radius para create_cylinder_points quando is_top=True
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

# --- FIM DAS FUNÇÕES DA CANECA ---


# --- CONFIGURAÇÕES DO TESTE DE ALINHAMENTO ---
# Parâmetros da caneca (iguais para ambas as nuvens)
CAN_OUTER_RADIUS = 0.04
CAN_INNER_RADIUS = 0.038
CAN_HEIGHT = 0.1
CAN_NUM_POINTS_BODY = 15000 # Reduzindo um pouco para o ICP ser mais rápido
CAN_NUM_POINTS_BOTTOM = 4000
CAN_NUM_POINTS_HANDLE = 6000

# Parâmetros do ICP
max_correspondence_distance = 0.005  # 5 mm - Distância máxima para considerar pontos como correspondentes
                                     # Importante: ajusta conforme a escala da sua nuvem.
                                     # Se for muito grande, pode pegar pontos errados.
                                     # Se for muito pequena, não encontrará correspondências.


# --- GERAÇÃO DAS NUVENS DE PONTOS ---
print("--- Iniciando o Teste de Alinhamento de Nuvens de Pontos ---")

# 1. Gerar a primeira nuvem de pontos (caneca "alvo" - target)
print("\n1. Gerando a primeira caneca (Alvo - verde)...")
pcd_target = generate_mug_pcd(CAN_OUTER_RADIUS, CAN_INNER_RADIUS, CAN_HEIGHT,
                              CAN_NUM_POINTS_BODY, CAN_NUM_POINTS_BOTTOM, CAN_NUM_POINTS_HANDLE)
pcd_target.paint_uniform_color([0.1, 0.7, 0.1]) # Pinta de verde

# 2. Gerar a segunda nuvem de pontos (caneca "fonte" - source)
#    E aplicamos uma transformação inicial (rotação e translação) para desalinhá-la
print("2. Gerando a segunda caneca (Fonte - azul) e aplicando um desalinhamento inicial...")
pcd_source = generate_mug_pcd(CAN_OUTER_RADIUS, CAN_INNER_RADIUS, CAN_HEIGHT,
                              CAN_NUM_POINTS_BODY, CAN_NUM_POINTS_BOTTOM, CAN_NUM_POINTS_HANDLE)
pcd_source.paint_uniform_color([0.1, 0.1, 0.7]) # Pinta de azul

# Criar uma transformação de desalinhamento (Rotação e Translação)
# Rotação de 5 graus no eixo Z e translação de 2 cm em X, 1 cm em Y e 3 cm em Z
theta_rot = np.deg2rad(5) # Rotação de 5 graus
R_initial = pcd_source.get_rotation_matrix_from_xyz((0, 0, theta_rot))
T_initial = np.array([0.02, 0.01, 0.03]) # Translação em X, Y, Z

initial_transform = np.eye(4) # Matriz identidade 4x4
initial_transform[:3, :3] = R_initial
initial_transform[:3, 3] = T_initial

# Aplicar a transformação inicial na nuvem fonte
pcd_source_transformed_initial = copy.deepcopy(pcd_source).transform(initial_transform)


# 3. Visualizar as Nuvens Desalinhadas
print("\n3. Visualizando as canecas DESALINHADAS (Alvo: Verde, Fonte: Azul). Feche para continuar...")
o3d.visualization.draw_geometries([pcd_target, pcd_source_transformed_initial],
                                  window_name="Canecas Desalinhadas",
                                  width=800, height=600)
print("   Visualização das canecas desalinhadas encerrada.")

# --- ALINHAMENTO USANDO ICP ---
print(f"\n4. Iniciando o processo de alinhamento ICP (Iterative Closest Point)...")
print(f"   Distância máxima de correspondência: {max_correspondence_distance} metros.")

# Defina a transformação inicial para o ICP.
# Aqui, usamos a transformação inversa para "tentar" trazê-la de volta ao alinhamento.
# No mundo real, você usaria métodos de registro global (RANSAC, FPFH) para ter uma boa inicialização.
# Para este teste sintético, a transformação inversa funciona bem para mostrar o ICP refinando.
# Ou, se não tivesse uma transformação inicial, o ICP tentaria a partir da identidade.
# Usar a transformação inversa do desalinhamento simula ter uma boa estimativa inicial.
initial_guess_for_icp = np.linalg.inv(initial_transform)


evaluation = o3d.pipelines.registration.evaluate_registration(
    pcd_source_transformed_initial, pcd_target, max_correspondence_distance, initial_guess_for_icp
)
print(f"   Avaliação inicial (antes do ICP): {evaluation}")


# Executar o ICP
# transformation_estimation_method: define como a transformação é calculada a cada iteração (ponto a ponto, ponto a plano)
# criteria: condições de parada do ICP (número máximo de iterações)
reg_p2p = o3d.pipelines.registration.registration_icp(
    pcd_source_transformed_initial, pcd_target,
    max_correspondence_distance,
    initial_guess_for_icp, # Ou np.eye(4) se não tiver uma estimativa inicial
    o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000) # Número máximo de iterações
)

print(f"   Resultado do Alinhamento ICP:")
print(f"   Transformação (matriz 4x4): \n{reg_p2p.transformation}")
print(f"   Inlier RMSE (Root Mean Square Error): {reg_p2p.inlier_rmse}") # Quão bem os pontos se encaixam
print(f"   Fitness (proporção de inliers): {reg_p2p.fitness}") # Quão grande é a sobreposição

# Aplicar a transformação final à nuvem fonte original para alinhá-la
pcd_source_aligned = copy.deepcopy(pcd_source).transform(reg_p2p.transformation)


# 5. Visualizar as Nuvens Alinhadas
print("\n5. Visualizando as canecas ALINHADAS (Alvo: Verde, Fonte Alinhada: Azul). Feche para continuar...")
# Agora visualizamos a nuvem alvo (verde) e a nuvem fonte JÁ ALINHADA (azul)
o3d.visualization.draw_geometries([pcd_target, pcd_source_aligned],
                                  window_name="Canecas Alinhadas",
                                  width=800, height=600)
print("   Visualização das canecas alinhadas encerrada.")

# --- COMBINAR AS NUVENS ALINHADAS ---
# Após o alinhamento, você geralmente combina as nuvens em uma só para formar um modelo completo.
print("\n6. Combinando as nuvens alinhadas em uma única nuvem de pontos...")
pcd_combined = pcd_target + pcd_source_aligned # Simplesmente soma as duas nuvens
print(f"   Nuvem combinada: {len(pcd_combined.points)} pontos.")

# Opcional: Aplicar downsampling e/ou remoção de ruído na nuvem combinada para limpeza final
print("   Aplicando downsampling na nuvem combinada para otimização...")
pcd_final_optimized = pcd_combined.voxel_down_sample(voxel_size=0.005) # Um voxel menor para detalhe
print(f"   Nuvem final otimizada: {len(pcd_final_optimized.points)} pontos.")

# 7. Visualizar a Nuvem Combinada e Otimizada
print("\n7. Visualizando a nuvem COMBINADA E OTIMIZADA. Feche para continuar...")
o3d.visualization.draw_geometries([pcd_final_optimized],
                                  window_name="Nuvem Combinada e Otimizada",
                                  width=800, height=600)
print("   Visualização da nuvem combinada e otimizada encerrada.")

# 8. Salvar o Resultado Final
output_combined_file = "caneca_combinada_e_otimizada.ply"
print(f"\n8. Salvando a nuvem combinada e otimizada em: {output_combined_file}")
o3d.io.write_point_cloud(output_combined_file, pcd_final_optimized)

print("\n--- Teste de Alinhamento de Nuvens de Pontos Concluído! ---")
print(f"O arquivo '{output_combined_file}' foi criado com a nuvem de pontos alinhada.")
