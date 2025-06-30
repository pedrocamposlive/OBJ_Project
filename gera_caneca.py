import open3d as o3d
import numpy as np
import os

# --- CONFIGURAÇÕES DA CANECA ---
# Corpo da caneca (cilindro oco)
radius_outer = 0.04  # Raio externo (em metros, ex: 4 cm)
radius_inner = 0.038 # Raio interno (para oco, ex: 3.8 cm)
height = 0.1         # Altura (em metros, ex: 10 cm)
num_points_body = 150000 # Número de pontos para o corpo

# Fundo da caneca (disco)
thickness_bottom = 0.005 # Espessura do fundo (ex: 0.5 cm)
num_points_bottom = 5000 # Número de pontos para o fundo

# Alça da caneca (simulando um toro ou arco)
handle_radius = 0.02   # Raio do arco da alça
handle_tube_radius = 0.005 # Espessura do "tubo" da alça
handle_center_offset = 0.05 # Deslocamento do centro da alça em relação ao centro da caneca
num_points_handle = 80000 # Número de pontos para a alça

# Nome do arquivo de saída
output_ply_file = "caneca_generica_DENSA.ply"

# --- FUNÇÕES AUXILIARES PARA GERAR PONTOS ---

def create_cylinder_points(radius, height, num_points, z_offset=0.0, is_top=False):
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
        for _ in range(num_points // 10): # Menos pontos para a borda
            theta = np.random.uniform(0, 2 * np.pi)
            x = np.random.uniform(radius_inner, radius_outer) * np.cos(theta)
            y = np.random.uniform(radius_inner, radius_outer) * np.sin(theta)
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
        # Ângulos aleatórios dentro do segmento desejado
        u = np.random.uniform(start_angle, end_angle)  # Ângulo ao redor do "corpo" do toro
        v = np.random.uniform(0, 2 * np.pi) # Ângulo ao redor do "tubo" do toro

        # Pontos do toro
        x = (major_radius + minor_radius * np.cos(v)) * np.cos(u) + center_offset_x
        y = (major_radius + minor_radius * np.cos(v)) * np.sin(u) + center_offset_y
        z = minor_radius * np.sin(v) + center_offset_z
        points.append([x, y, z])
    return np.array(points)

# --- GERAÇÃO DA NUVEM DE PONTOS DA CANECA ---

print(f"Gerando pontos para o corpo externo da caneca...")
# Corpo externo do cilindro (com borda superior)
pcd_outer_body = create_cylinder_points(radius_outer, height, num_points_body // 2, is_top=True)
pcd_outer_body_top = create_disk_points(radius_outer, num_points_body // 10, height) # Borda superior externa

print(f"Gerando pontos para o corpo interno da caneca...")
# Corpo interno do cilindro (inverter normais na reconstrução, mas aqui só pontos)
pcd_inner_body = create_cylinder_points(radius_inner, height, num_points_body // 2)

print(f"Gerando pontos para o fundo da caneca...")
# Fundo da caneca (disco)
pcd_bottom_top = create_disk_points(radius_outer, num_points_bottom, thickness_bottom) # Parte superior do fundo
pcd_bottom_bottom = create_disk_points(radius_outer, num_points_bottom, 0.0) # Parte inferior do fundo

# Alça da caneca
print(f"Gerando pontos para a alça da caneca...")
# Ajusta a posição da alça para o lado da caneca e na altura certa
# O centro da alça estará no meio da altura da caneca, deslocado para o lado
handle_offset_y = -(radius_outer + handle_radius * 0.8) # Move para fora da caneca
handle_offset_z = height / 2 # Meio da altura

# Usaremos dois segmentos de toro para simular a alça (superior e inferior)
# Estes são "arcos" que simulam a forma da alça
# O ângulo de 0 a pi/2 é o primeiro quarto de círculo (topo)
# O ângulo de pi/2 a pi é o segundo quarto (inferior)
pcd_handle_top_arc = create_torus_segment_points(
    major_radius=handle_radius, minor_radius=handle_tube_radius,
    num_points=num_points_handle // 2,
    start_angle=np.pi * 0.05, end_angle=np.pi * 0.5, # Pequeno ajuste para evitar pontas fechadas
    center_offset_x=0.0, center_offset_y=handle_offset_y, center_offset_z=handle_offset_z
)

pcd_handle_bottom_arc = create_torus_segment_points(
    major_radius=handle_radius, minor_radius=handle_tube_radius,
    num_points=num_points_handle // 2,
    start_angle=np.pi * 0.5, end_angle=np.pi * 0.95, # Ajuste para completar a parte de baixo
    center_offset_x=0.0, center_offset_y=handle_offset_y, center_offset_z=handle_offset_z
)

# Inverter X para a alça ficar do lado certo (se a caneca for centrada no Y)
# Ou rotacionar os pontos do corpo da caneca para ter a alça no +X
# Para simplicidade, vamos mover a alça para o lado positivo de X
# (Assumindo a caneca centrada no eixo Z)
handle_rotation_angle = np.pi / 2 # Rotaciona 90 graus para frente (eixo X)
rotation_matrix = np.array([
    [np.cos(handle_rotation_angle), -np.sin(handle_rotation_angle), 0],
    [np.sin(handle_rotation_angle),  np.cos(handle_rotation_angle), 0],
    [0, 0, 1]
])

pcd_handle_top_arc_rotated = (rotation_matrix @ pcd_handle_top_arc.T).T
pcd_handle_bottom_arc_rotated = (rotation_matrix @ pcd_handle_bottom_arc.T).T


# Combinar todas as partes em uma única nuvem de pontos
all_points = np.vstack([
    pcd_outer_body,
    pcd_inner_body,
    pcd_bottom_top,
    pcd_bottom_bottom,
    pcd_handle_top_arc_rotated,
    pcd_handle_bottom_arc_rotated
])

# Criar o objeto PointCloud do Open3D
pcd_caneca = o3d.geometry.PointCloud()
pcd_caneca.points = o3d.utility.Vector3dVector(all_points)

# Opcional: Adicionar cores aos pontos (para melhor visualização)
# Vamos pintar todos os pontos de um marrom/laranja
colors = np.tile(np.array([0.7, 0.4, 0.1]), (len(all_points), 1)) # RGB para marrom/laranja
pcd_caneca.colors = o3d.utility.Vector3dVector(colors)


# --- SALVAR E VISUALIZAR ---
print(f"\nTotal de pontos gerados para a caneca: {len(pcd_caneca.points)}.")

# Salvar a nuvem de pontos em formato .ply
print(f"Salvando a caneca gerada em: {output_ply_file}")
o3d.io.write_point_cloud(output_ply_file, pcd_caneca)

# Visualizar a caneca gerada
print(f"\nVisualizando a caneca gerada. Feche a janela para continuar.")
o3d.visualization.draw_geometries([pcd_caneca],
                                  window_name="Caneca Genérica Gerada",
                                  width=800, height=600)
print("Visualização encerrada.")
print(f"\nArquivo '{output_ply_file}' criado com sucesso!")
