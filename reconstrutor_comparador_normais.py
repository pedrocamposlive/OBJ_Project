import open3d as o3d
import numpy as np
import os
import copy # Para duplicar objetos

# --- CONFIGURAÇÕES ---
# 1. Nome do arquivo de entrada (usar a caneca densa que você gerou)
input_pcd_file = "caneca_generica_DENSA.ply" 

# 2. Parâmetros Comuns de Reconstrução
POISSON_DEPTH = 10 # Nível de detalhe da reconstrução de malha (ajuste conforme necessário)
VOXEL_SIZE_PCD_FOR_RECONSTRUCTION = 0.005 # Downsampling antes da reconstrução (5 mm)

# 3. Parâmetros para Estimativa de Normais (para o método HYBRID)
#    - search_radius: Raio de busca para vizinhos.
#    - max_nn: Número máximo de vizinhos.
HYBRID_NORMAL_SEARCH_RADIUS = 0.15 # 15 cm
HYBRID_NORMAL_MAX_NEIGHBORS = 50

# 4. Parâmetros para Estimativa de Normais (para o método KNN)
#    - k: Número fixo de vizinhos mais próximos a considerar.
KNN_NORMAL_K_NEIGHBORS = 30 # Tente 20, 30 ou 40.

# 5. Escolha o método de estimativa de normais: "HYBRID" ou "KNN"
NORMAL_ESTIMATION_METHOD = "HYBRID" # Altere para "KNN" para comparar


# --- FUNÇÕES AUXILIARES ---

def estimate_and_orient_normals(pcd_input, method="HYBRID", radius=None, max_nn=None, k=None):
    """Estima e orienta as normais da nuvem de pontos."""
    pcd = copy.deepcopy(pcd_input) # Trabalhar com uma cópia
    print(f"   Estimando as normais usando o método: {method}...")

    if method == "HYBRID":
        if radius is None or max_nn is None:
            raise ValueError("Radius and max_nn must be provided for HYBRID method.")
        search_param = o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn)
    elif method == "KNN":
        if k is None:
            raise ValueError("k (number of neighbors) must be provided for KNN method.")
        search_param = o3d.geometry.KDTreeSearchParamKNN(k=k)
    else:
        raise ValueError("Método de estimativa de normais inválido. Escolha 'HYBRID' ou 'KNN'.")
    
    pcd.estimate_normals(search_param=search_param)
    
    # Orientar as normais para apontar para uma direção consistente (muito importante para Poisson)
    pcd.orient_normals_to_align_with_direction(orientation_reference=np.array([0., 0., 1.])) # Aponta para o eixo Z positivo
    
    return pcd


# --- FLUXO PRINCIPAL DO TESTE ---

print("--- Iniciando o Teste de Comparação de Estimativa de Normais ---")

# 0. Verificar a existência do arquivo de entrada
if not os.path.exists(input_pcd_file):
    print(f"ERRO: O arquivo '{input_pcd_file}' não foi encontrado.")
    print("Por favor, verifique o nome do arquivo e o caminho.")
    exit()

# 1. Carregar a Nuvem de Pontos
print(f"\n1. Carregando a nuvem de pontos: {input_pcd_file}")
pcd_original = o3d.io.read_point_cloud(input_pcd_file)
print(f"   Nuvem de pontos carregada: {len(pcd_original.points)} pontos.")

# Aplicar downsampling para otimização antes da estimativa de normais e reconstrução
print(f"   Aplicando downsampling (Voxel Grid) com voxel_size: {VOXEL_SIZE_PCD_FOR_RECONSTRUCTION}m...")
pcd_for_reconstruction = pcd_original.voxel_down_sample(voxel_size=VOXEL_SIZE_PCD_FOR_RECONSTRUCTION) 
print(f"   Nuvem após downsampling (para reconstrução): {len(pcd_for_reconstruction.points)} pontos.")


# 2. Estimando as Normais da Superfície (com o método escolhido)
if NORMAL_ESTIMATION_METHOD == "HYBRID":
    pcd_with_normals = estimate_and_orient_normals(pcd_for_reconstruction, 
                                                 method="HYBRID", 
                                                 radius=HYBRID_NORMAL_SEARCH_RADIUS, 
                                                 max_nn=HYBRID_NORMAL_MAX_NEIGHBORS)
elif NORMAL_ESTIMATION_METHOD == "KNN":
    pcd_with_normals = estimate_and_orient_normals(pcd_for_reconstruction, 
                                                 method="KNN", 
                                                 k=KNN_NORMAL_K_NEIGHBORS)

# 3. Reconstruir a Superfície (Mesh) usando o algoritmo de Poisson
print(f"\n3. Reconstruindo a superfície (mesh) usando o algoritmo de Poisson com profundidade (depth): {POISSON_DEPTH}...")
mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd_with_normals, depth=POISSON_DEPTH)

# Limpar o mesh resultante (cortar pela bounding box)
bbox = pcd_with_normals.get_axis_aligned_bounding_box()
mesh = mesh.crop(bbox)

# Opcional: Suavizar a Malha (pode ajudar na aparência)
# print("\n4. Suavizando a malha (Opcional)...")
# mesh = mesh.filter_smooth_laplacian(number_of_iterations=5) 
# mesh.compute_vertex_normals() 


# 4. Visualizar a Malha Reconstruída
output_mesh_file = f"caneca_reconstruida_normals_{NORMAL_ESTIMATION_METHOD}.obj"
print(f"\n4. Visualizando a malha reconstruída. Feche a janela para continuar...")
o3d.visualization.draw_geometries([mesh],
                                  window_name=f"Malha Reconstruída ({NORMAL_ESTIMATION_METHOD} Normals)",
                                  width=800, height=600)
print("   Visualização encerrada.")

# 5. Salvar a Malha Reconstruída
print(f"\n5. Salvando a malha reconstruída em: {output_mesh_file}")
o3d.io.write_triangle_mesh(output_mesh_file, mesh)
print(f"   O arquivo '{output_mesh_file}' foi criado com a malha 3D.")

print("\n--- Teste de Comparação de Estimativa de Normais Concluído! ---")
