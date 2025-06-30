import open3d as o3d
import numpy as np
import os
import copy


# --- CONFIGURAÇÕES ---
# 1. Nome do arquivo de entrada (usar a caneca densa)
input_pcd_file = "caneca_generica_DENSA.ply" 

# 2. Nome do arquivo de saída para a malha
output_mesh_file = "caneca_reconstruida_ROBUSTA.obj" # Novo nome para o arquivo

# 3. Parâmetros para Estimativa de Normais
# Ajustar o raio para a nova densidade.
normal_search_radius = 0.15 # Voltei para 0.15m (15cm) - pode ser mais robusto
normal_max_neighbors = 50   # Mantendo em 50

# 4. Parâmetros para Reconstrução de Superfície (Poisson)
poisson_depth = 10          # Reduzido de 11 para 10 - mais robusto, menos detalhe extremo

# --- INÍCIO DO SCRIPT ---
print("--- Iniciando a Reconstrução de Superfície (Mesh) ---")

# 0. Verificar a existência do arquivo de entrada
if not os.path.exists(input_pcd_file):
    print(f"ERRO: O arquivo '{input_pcd_file}' não foi encontrado.")
    print("Por favor, verifique o nome do arquivo e o caminho.")
    exit()

# 1. Carregar a Nuvem de Pontos
print(f"\n1. Carregando a nuvem de pontos: {input_pcd_file}")
pcd = o3d.io.read_point_cloud(input_pcd_file)
print(f"   Nuvem de pontos carregada: {len(pcd.points)} pontos.")

# **IMPORTANTE**: Aplicar um DOWNSEMPLING para a reconstrução, mesmo que seja densa.
#                 Isso ajuda a uniformizar e reduzir o ruído residual para o Poisson.
#                 Um voxel_size muito pequeno para manter detalhes.
print("   Aplicando downsampling (Voxel Grid) antes da reconstrução para otimização e uniformização...")
# Experimente com 0.005 ou 0.01. Para 0.005, a caneca de 4cm de raio é bem densa.
pcd = pcd.voxel_down_sample(voxel_size=0.005) 
print(f"   Nuvem após downsampling (para reconstrução): {len(pcd.points)} pontos.")

# **OPCIONAL**: Aplicar um filtro de ruído ligeiro novamente, se a nuvem ainda tiver muitos pontos errados
# print("   Aplicando filtro de ruído (SOR) antes da reconstrução...")
# cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.5) # std_ratio um pouco mais tolerante
# pcd = cl
# print(f"   Nuvem após filtro SOR (opcional): {len(pcd.points)} pontos.")


# 2. Estimar Normais da Superfície
print(f"\n2. Estimando as normais da superfície com raio {normal_search_radius} e {normal_max_neighbors} vizinhos...")
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
    radius=normal_search_radius, max_nn=normal_max_neighbors))

# Orientar as normais para apontar para uma direção consistente (muito importante para Poisson)
pcd.orient_normals_to_align_with_direction(orientation_reference=np.array([0., 0., 1.])) # Aponta para o eixo Z positivo

# 3. Reconstruir a Superfície (Mesh) usando o algoritmo de Poisson
print(f"\n3. Reconstruindo a superfície (mesh) usando o algoritmo de Poisson com profundidade (depth): {poisson_depth}...")
mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=poisson_depth)

# Opcional: Remover vértices com baixa densidade para limpar o mesh resultante
# densities = np.asarray(densities)
# mesh = mesh.select_by_index(np.where(densities > np.quantile(densities, 0.05))[0]) # Remove os 5% de vértices menos densos

# Cortar o mesh para a caixa delimitadora da nuvem original (ajuda a remover "fantasmas" longe do objeto)
bbox = pcd.get_axis_aligned_bounding_box()
mesh = mesh.crop(bbox)


# 4. Suavizar a Malha (Opcional - mas pode ajudar na aparência)
print("\n4. Suavizando a malha (Opcional - pode levar tempo)...")
# Número de iterações de suavização. Mais iterações = mais suave, mas pode perder detalhes.
mesh = mesh.filter_smooth_laplacian(number_of_iterations=5) 
mesh.compute_vertex_normals() # Recomputar normais após suavização para renderização correta

# 5. Visualizar a Malha Reconstruída
print(f"\n5. Visualizando a malha reconstruída. Feche a janela para continuar...")
o3d.visualization.draw_geometries([mesh],
                                  window_name="Malha Reconstruída Robusta",
                                  width=800, height=600)
print("   Visualização encerrada.")

# 6. Salvar a Malha Reconstruída
print(f"\n6. Salvando a malha reconstruída em: {output_mesh_file}")
o3d.io.write_triangle_mesh(output_mesh_file, mesh)

print("\n--- Reconstrução de Superfície Concluída! ---")
print(f"O arquivo '{output_mesh_file}' foi criado com a malha 3D.")