import open3d as o3d
import numpy as np
import copy
import os

# --- CONFIGURAÇÕES ---
# Nome do arquivo base que usaremos para gerar múltiplos fragments
base_fragment_file = "fragment.ply" 

# Nomes dos arquivos PLY para salvar cada fragmento transformado (simulando capturas diferentes)
output_fragment_1 = "fragment_simulado_01.ply"
output_fragment_2 = "fragment_simulado_02.ply"
output_fragment_3 = "fragment_simulado_03.ply"

# Parâmetros de pré-processamento (para o fragment.ply)
voxel_size = 0.05 # 5 cm - para downsampling antes do registro global (FPFH é sensível à densidade)
max_correspondence_distance_fine = 0.005 # 5 mm para ICP (registro fino)

# --- FUNÇÕES AUXILIARES PARA REGISTRO ---

def draw_registration_result(source, target, transformation):
    """Função para visualizar o resultado do registro."""
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0]) # Laranja
    target_temp.paint_uniform_color([0, 0.651, 0.929]) # Azul
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp],
                                      window_name="Registro",
                                      width=800, height=600)

def preprocess_point_cloud(pcd, voxel_size):
    """Aplica downsampling e estima normais para o registro."""
    print(f"   Downsampling com voxel size {voxel_size}...")
    pcd_down = pcd.voxel_down_sample(voxel_size)
    
    print("   Estimando normais...")
    # Raio e vizinhos para estimar normais (ajustados para a nuvem downsampled)
    radius_normal = voxel_size * 2
    pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    
    print("   Calculando FPFH features...")
    # Raio para calcular features FPFH (geralmente 5x o voxel_size)
    radius_feature = voxel_size * 5
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(pcd_down,
                                                            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh


def execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size):
    """Executa o registro global (FPFH + RANSAC)."""
    distance_threshold = voxel_size * 1.5 # Limiar de distância para correspondência
    print(f"   Executando registro global com RANSAC (limiar: {distance_threshold})...")
    
    # CORREÇÃO AQUI: Adicionar o argumento 'ransac_n=3'
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True, # True é mutual_filter
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=3, # <<< ADICIONADO ESTE ARGUMENTO
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999) # O 'criteria' também é um argumento nomeado
    )
    return result


def refine_registration(source, target, source_fpfh, target_fpfh, initial_transform, voxel_size):
    """Executa o registro fino (ICP)."""
    distance_threshold = voxel_size * 0.4 # Limiar para ICP (mais apertado)
    print(f"   Refinando registro com ICP (limiar: {distance_threshold})...")
    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, initial_transform,
        o3d.pipelines.registration.TransformationEstimationPointToPoint()) # PointToPlane pode ser melhor para planar surfaces
    return result

# --- INÍCIO DO SCRIPT ---
print("--- Iniciando o Teste de Registro Global de Múltiplos Fragments ---")

# 0. Verificar a existência do arquivo base
if not os.path.exists(base_fragment_file):
    print(f"ERRO: O arquivo base '{base_fragment_file}' não foi encontrado.")
    print("Por favor, baixe 'fragment.ply' do Open3D (link fornecido anteriormente) e coloque-o na mesma pasta.")
    exit()

# 1. Carregar o fragmento base
print(f"\n1. Carregando o fragmento base: {base_fragment_file}")
base_pcd = o3d.io.read_point_cloud(base_fragment_file)
print(f"   Fragmento base: {len(base_pcd.points)} pontos.")


# 2. Gerar múltiplos fragments simulados com transformações conhecidas
print("\n2. Gerando fragments simulados com transformações...")
# Fragmento 1 (target) - ligeiramente movido para ter um centro mais "útil"
pcd1_orig = copy.deepcopy(base_pcd).translate([-0.5, 0, 0])
pcd1_orig.paint_uniform_color([0.8, 0.1, 0.1]) # Vermelho
o3d.io.write_point_cloud(output_fragment_1, pcd1_orig)

# Fragmento 2 (source) - original, mas será alinhado ao 1
# CORREÇÃO AQUI: Dividir as operações
pcd2_orig = copy.deepcopy(base_pcd)
pcd2_orig.translate([0.5, 0, 0.1])
pcd2_orig.rotate(pcd2_orig.get_rotation_matrix_from_xyz((0.1, 0, 0)))
pcd2_orig.paint_uniform_color([0.1, 0.8, 0.1]) # Verde
o3d.io.write_point_cloud(output_fragment_2, pcd2_orig)

# Fragmento 3 (source) - desalinhado ainda mais
# CORREÇÃO AQUI: Dividir as operações
pcd3_orig = copy.deepcopy(base_pcd)
pcd3_orig.translate([0, 0.5, -0.1])
pcd3_orig.rotate(pcd3_orig.get_rotation_matrix_from_xyz((0, 0.2, 0)))
pcd3_orig.paint_uniform_color([0.1, 0.1, 0.8]) # Azul
o3d.io.write_point_cloud(output_fragment_3, pcd3_orig)

# Criar uma lista das nuvens de pontos a serem alinhadas
pcds_to_align = [pcd1_orig, pcd2_orig, pcd3_orig]
print(f"   {len(pcds_to_align)} fragments simulados gerados e salvos.")

# 3. Visualizar todos os fragments desalinhados
print("\n3. Visualizando todos os fragments DESALINHADOS. Feche para continuar...")
o3d.visualization.draw_geometries(pcds_to_align,
                                  window_name="Fragments Desalinhados",
                                  width=800, height=600, left=50, top=50)
print("   Visualização encerrada.")


# 4. Preparar cada nuvem de pontos para registro (downsample e features)
print("\n4. Pré-processando cada fragmento para registro (downsampling, normais, FPFH)...")
pcds_down = []
pcds_fpfh = []
for i, pcd in enumerate(pcds_to_align):
    print(f"   Processando fragmento {i+1}...")
    pcd_down, pcd_fpfh = preprocess_point_cloud(pcd, voxel_size)
    pcds_down.append(pcd_down)
    pcds_fpfh.append(pcd_fpfh)


# 5. Executar o Registro Global Sequencialmente
print("\n5. Iniciando Registro Global Sequencial (FPFH + RANSAC + ICP Fino)...")

# O primeiro fragmento será nosso TARGET de referência
full_pcd_aligned = copy.deepcopy(pcds_down[0])
full_pcd_aligned_orig_res = copy.deepcopy(pcds_to_align[0]) # Manter a resolução original para o final

# Matriz de transformação acumulada (relativa ao primeiro fragmento)
# Transfomações relativas entre o target e o source
# Esta lista irá armazenar as transformações que levam CADA fragmento à sua posição final alinhada
transformations = [np.eye(4)] # A primeira transformação é identidade (para o pcd1_orig)

# Alinhar os fragmentos seguintes ao primeiro
for i in range(len(pcds_down) - 1):
    source_down = pcds_down[i+1] # Fragmento a ser alinhado
    target_down = pcds_down[0]   # Fragmento de referência (o primeiro)
    
    print(f"\n   Alinhando fragmento {i+2} (source) ao fragmento 1 (target)...")
    
    # Registro Global (Rough alignment with FPFH)
    result_global = execute_global_registration(source_down, target_down, 
                                                pcds_fpfh[i+1], pcds_fpfh[0], voxel_size)
    print(f"     Resultado Global: Fitness={result_global.fitness:.4f}, Inlier RMSE={result_global.inlier_rmse:.4f}")
    
    # Refinamento com ICP (Fine alignment)
    result_fine = refine_registration(source_down, target_down, 
                                      pcds_fpfh[i+1], pcds_fpfh[0], 
                                      result_global.transformation, voxel_size)
    print(f"     Resultado Fino (ICP): Fitness={result_fine.fitness:.4f}, Inlier RMSE={result_fine.inlier_rmse:.4f}")

    # Armazena a transformação final para este fragmento
    transformations.append(result_fine.transformation) 


# 6. Combinar todas as nuvens de pontos na resolução original usando as transformações
print("\n6. Combinando todas as nuvens de pontos na resolução original usando as transformações...")

pcd_combined_final = o3d.geometry.PointCloud()
for i, pcd_orig in enumerate(pcds_to_align):
    # Transforma cada nuvem original usando sua transformação acumulada
    pcd_transformed = copy.deepcopy(pcd_orig).transform(transformations[i])
    pcd_combined_final += pcd_transformed # Adiciona ao combinado

print(f"   Nuvem de pontos combinada final: {len(pcd_combined_final.points)} pontos.")

# 7. Visualizar a Nuvem de Pontos Combinada Final
print("\n7. Visualizando a nuvem COMBINADA FINAL. Feche para continuar...")
o3d.visualization.draw_geometries([pcd_combined_final],
                                  window_name="Nuvem Combinada Final",
                                  width=800, height=600, left=900, top=50)
print("   Visualização encerrada.")

# Opcional: Aplicar um downsampling final na nuvem combinada se for muito grande
print("   Aplicando downsampling final na nuvem combinada para otimização...")
final_downsampled_pcd = pcd_combined_final.voxel_down_sample(voxel_size=0.01) # 1 cm
print(f"   Nuvem final otimizada para exportação: {len(final_downsampled_pcd.points)} pontos.")

# 8. Salvar o Resultado Final
output_combined_file = "fragmentos_combinados_e_alinhados.ply"
print(f"\n8. Salvando a nuvem combinada e otimizada em: {output_combined_file}")
o3d.io.write_point_cloud(output_combined_file, final_downsampled_pcd)

print("\n--- Teste de Registro Global de Múltiplos Fragments Concluído! ---")
