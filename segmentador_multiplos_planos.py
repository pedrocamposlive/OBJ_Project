import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt # Para visualização de cores
import os
import copy

# --- CONFIGURAÇÕES ---
# 1. Parâmetros para Filtro Estatístico (SOR) - para nuvens de pontos reais
sor_nb_neighbors = 20
sor_std_ratio = 2.0

# 2. Parâmetros para Downsampling (Voxel Grid)
voxel_grid_size = 0.02 # 2 cm

# 3. Parâmetros para Detecção de Plano (RANSAC)
# distance_threshold: Distância máxima de um ponto ao plano para ser inlier.
#                    Ajuste conforme a precisão da sua captura e a rugosidade da superfície.
ransac_distance_threshold = 0.03 # 3 cm, um pouco maior para ambientes reais
ransac_n = 3                     # Número de pontos para estimar o plano
ransac_num_iterations = 1000     # Número de iterações do RANSAC

# 4. Número máximo de planos para detectar e remover
max_planes_to_remove = 3 # Tente remover o chão e 2 paredes, por exemplo

# 5. Parâmetros para Clustering (DBScan) nos objetos restantes
# eps: Raio de busca para vizinhos. Pontos dentro deste raio são conectados.
# min_points: Número mínimo de pontos para formar um cluster.
dbscan_eps = 0.1  # 10 cm - pode precisar ser maior para objetos maiores ou mais esparsos
dbscan_min_points = 50 # Mínimo de pontos para um objeto significativo

# --- INÍCIO DO SCRIPT ---
print("--- Iniciando o Teste de Segmentação de Múltiplos Planos ---")

# 1. Carregar uma Nuvem de Pontos de Exemplo (Dataset do Open3D)
# Este é um dataset de exemplo que vem com o Open3D.
# Se você tiver um PLY de um ambiente interno real, pode usar aqui!
print("\n1. Carregando nuvem de pontos de exemplo (sala ou ambiente) 'fragment.ply'...")
# O Open3D tem alguns datasets embutidos. 'fragment.ply' é bom para isso.
# Ou você pode baixar do Open3D's tutorial data: https://github.com/isl-org/Open3D/tree/main/examples/test_data/Fragment
# Por simplicidade, vamos usar um que pode ser carregado localmente se você o baixar.
# Se preferir, pode baixar um PLY de ambiente de Kaggle ou SketchUp e colocar na pasta.
try:
    pcd_path = "fragment.ply" # Nome do arquivo que você deve baixar e colocar na pasta
    if not os.path.exists(pcd_path):
        print(f"ATENÇÃO: '{pcd_path}' não encontrado. Tentando carregar um dataset genérico do Open3D...")
        # Se você tiver problemas para baixar, pode tentar carregar um dataset de teste menor do Open3D,
        # mas eles não são de ambientes. Para simular ambiente, baixar 'fragment.ply' é melhor.
        # Ou um PLY de ambiente de https://help.sketchup.com/en/scan-essentials-sketchup/sample-point-cloud-data
        raise FileNotFoundError # Força a exceção para instruir o usuário
    pcd_original = o3d.io.read_point_cloud(pcd_path)
    if not pcd_original.has_points():
        raise ValueError("Nuvem de pontos carregada está vazia.")
except (FileNotFoundError, ValueError):
    print("Por favor, baixe o arquivo 'fragment.ply' do Open3D:")
    print("Visite: https://github.com/isl-org/Open3D/tree/main/examples/test_data/Fragment")
    print("Salve 'fragment.ply' na mesma pasta do script.")
    print("Alternativamente, use um dos links do Polycam/SketchUp para um .PLY de ambiente real.")
    exit()

print(f"   Nuvem de pontos original: {len(pcd_original.points)} pontos.")

# 2. Visualizar a Nuvem de Pontos Original
print("\n2. Visualizando a nuvem ORIGINAL. Feche a janela para continuar...")
o3d.visualization.draw_geometries([pcd_original],
                                  window_name="Nuvem Original",
                                  width=800, height=600, left=50, top=50)
print("   Visualização da nuvem original encerrada.")

# 3. Pré-processamento: Remover Ruído e Downsampling
print(f"\n3. Pré-processamento: Removendo ruído (SOR) e aplicando downsampling (Voxel Grid)...")
pcd_processed = pcd_original.voxel_down_sample(voxel_size=voxel_grid_size)
cl, ind = pcd_processed.remove_statistical_outlier(nb_neighbors=sor_nb_neighbors, std_ratio=sor_std_ratio)
pcd_processed = cl
print(f"   Nuvem de pontos após pré-processamento: {len(pcd_processed.points)} pontos.")


# 4. Detecção e Remoção ITERATIVA de Múltiplos Planos
print(f"\n4. Iniciando detecção e remoção ITERATIVA de planos (chão, paredes, teto)...")

pcd_remaining = copy.deepcopy(pcd_processed) # Começa com a nuvem processada
planes = [] # Para armazenar os planos detectados
for i in range(max_planes_to_remove):
    print(f"   Tentativa {i+1}/{max_planes_to_remove}: Detectando plano...")
    model, inliers = pcd_remaining.segment_plane(distance_threshold=ransac_distance_threshold,
                                               ransac_n=ransac_n,
                                               num_iterations=ransac_num_iterations)
    
    if len(inliers) < 100: # Se o plano detectado for muito pequeno, parar
        print(f"     Plano muito pequeno ou nenhum plano significativo encontrado na tentativa {i+1}. Parando.")
        break

    plane_pcd = pcd_remaining.select_by_index(inliers)
    plane_pcd.paint_uniform_color([np.random.rand(), np.random.rand(), np.random.rand()]) # Cor aleatória para cada plano
    planes.append(plane_pcd)
    
    # Remover o plano detectado da nuvem e continuar com o restante
    pcd_remaining = pcd_remaining.select_by_index(inliers, invert=True)
    print(f"     Plano detectado com {len(inliers)} pontos. Pontos restantes: {len(pcd_remaining.points)}")

# 5. Visualizar os Planos Removidos e a Nuvem Restante (Objetos)
print("\n5. Visualizando os PLANOS DETECTADOS (coloridos) e o restante da nuvem (Objetos). Feche para continuar...")
o3d.visualization.draw_geometries(planes + [pcd_remaining.paint_uniform_color([0.8, 0.2, 0.8])], # Restante em roxo
                                  window_name="Planos Removidos e Objetos Restantes",
                                  width=800, height=600, left=50, top=400)
print("   Visualização encerrada.")


# 6. Agrupamento (Clustering) para Isolar Objetos Restantes
print(f"\n6. Aplicando Clustering (DBScan) nos objetos restantes com:")
print(f"   - Raio (eps): {dbscan_eps} m")
print(f"   - Mínimo de pontos por cluster: {dbscan_min_points}")

labels = np.array(pcd_remaining.cluster_dbscan(eps=dbscan_eps, min_points=dbscan_min_points, print_progress=True))

max_label = labels.max()
print(f"   Número total de clusters detectados: {max_label + 1} (incluindo ruído -1).")

# Visualizar os clusters com cores diferentes
colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
colors[labels < 0] = 0 # Pontos de ruído (não clusterizados) em preto
pcd_remaining.colors = o3d.utility.Vector3dVector(colors[:, :3])

print("\n7. Visualizando os OBJETOS CLUSTERIZADOS. Feche a janela para continuar...")
o3d.visualization.draw_geometries([pcd_remaining],
                                  window_name="Objetos Clusterizados",
                                  width=800, height=600, left=900, top=400)
print("   Visualização encerrada.")

# 8. Opcional: Salvar cada cluster como um arquivo PLY separado
output_folder = "objetos_isolados"
os.makedirs(output_folder, exist_ok=True) # Cria a pasta se não existir

isolated_objects_count = 0
if max_label >= 0:
    for i in range(max_label + 1):
        if i == -1: # Ignorar ruído
            continue
        cluster_pcd = pcd_remaining.select_by_index(np.where(labels == i)[0])
        if len(cluster_pcd.points) >= dbscan_min_points: # Salvar apenas clusters significativos
            output_file = os.path.join(output_folder, f"objeto_cluster_{i}.ply")
            o3d.io.write_point_cloud(output_file, cluster_pcd)
            print(f"   Cluster {i} salvo em: {output_file} ({len(cluster_pcd.points)} pontos)")
            isolated_objects_count += 1
else:
    print("   Nenhum cluster válido (com pontos suficientes) foi detectado.")

print(f"\nTotal de {isolated_objects_count} objetos isolados salvos na pasta '{output_folder}'.")
print("\n--- Teste de Segmentação de Múltiplos Planos Concluído! ---")
