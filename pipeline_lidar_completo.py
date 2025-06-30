import open3d as o3d
import numpy as np
import copy
import os
import matplotlib.pyplot as plt # Necessário para colorir clusters

# --- CONFIGURAÇÕES GLOBAIS ---
# Caminho para o fragmento base (fragment.ply deve estar na mesma pasta)
BASE_FRAGMENT_FILE = "fragment.ply"

# Parâmetros de pré-processamento (gerais para todo o pipeline)
VOXEL_SIZE_GLOBAL_REG = 0.05 # Voxel size para o downsampling antes do registro global (FPFH)
VOXEL_SIZE_FINAL_PCD = 0.01  # Voxel size para o downsampling da nuvem combinada final (1 cm)
VOXEL_SIZE_OBJ_RECONSTRUCTION = 0.005 # Voxel size para objetos individuais antes da reconstrução (5 mm)

# Parâmetros para Filtro Estatístico (SOR)
SOR_NB_NEIGHBORS = 20
SOR_STD_RATIO = 2.0

# Parâmetros para Detecção de Plano (RANSAC)
RANSAC_DISTANCE_THRESHOLD = 0.03 # 3 cm
RANSAC_N = 3
RANSAC_NUM_ITERATIONS = 1000
MAX_PLANES_TO_REMOVE = 3 # Ex: Chão, duas paredes principais

# Parâmetros para Clustering (DBScan)
DBSCAN_EPS = 0.1         # Raio de busca para vizinhos (10 cm)
DBSCAN_MIN_POINTS = 50   # Mínimo de pontos para formar um cluster (considerar um objeto)

# Parâmetros para Reconstrução de Superfície (Poisson)
NORMAL_SEARCH_RADIUS = 0.15 # Raio para estimar normais (15 cm)
NORMAL_MAX_NEIGHBORS = 50
POISSON_DEPTH = 10 # Nível de detalhe da reconstrução de malha

# --- FUNÇÕES AUXILIARES ---

def preprocess_point_cloud(pcd, voxel_size_down):
    """Aplica downsampling e estima normais e FPFH features."""
    pcd_down = pcd.voxel_down_sample(voxel_size_down)
    radius_normal = voxel_size_down * 2
    pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    radius_feature = voxel_size_down * 5
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh

def execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size_reg):
    """Executa o registro global (FPFH + RANSAC)."""
    distance_threshold = voxel_size_reg * 1.5
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=3, # Explicitly pass ransac_n
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    return result

def refine_registration(source, target, initial_transform, voxel_size_reg):
    """Executa o registro fino (ICP)."""
    distance_threshold = voxel_size_reg * 0.4
    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, initial_transform,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    return result

# --- FLUXO PRINCIPAL DO ROBÔ ---

def run_full_pipeline(input_fragments_list):
    """
    Orquestra o pipeline completo de processamento de nuvens de pontos.
    input_fragments_list: Lista de caminhos para os arquivos .ply de varreduras.
    """
    print("\n--- PASSO 1: Carregar e Pré-processar Fragmentos ---")
    pcds_original_res = [] # Armazena PCDS na resolução original para o merge final
    pcds_down = [] # Armazena PCDS downsampled para registro
    pcds_fpfh = [] # Armazena features para registro

    for i, file_path in enumerate(input_fragments_list):
        print(f"   Carregando e pré-processando fragmento {i+1}: {file_path}")
        pcd_orig = o3d.io.read_point_cloud(file_path)
        if not pcd_orig.has_points():
            print(f"   Aviso: Fragmento {file_path} está vazio. Ignorando.")
            continue
        pcds_original_res.append(pcd_orig)
        
        # Downsample e FPFH para registro
        pcd_d, pcd_f = preprocess_point_cloud(pcd_orig, VOXEL_SIZE_GLOBAL_REG)
        pcds_down.append(pcd_d)
        pcds_fpfh.append(pcd_f)

    if not pcds_down:
        print("Erro: Nenhuma nuvem de pontos válida para processar.")
        return None, None, None


    print("\n--- PASSO 2: Registro Global e Alinhamento (FPFH + RANSAC + ICP) ---")
    # Usa o primeiro fragmento como referência (target)
    # Lista para armazenar as transformações que alinham CADA fragmento ao primeiro
    transformations = [np.eye(4)] # Transformação do primeiro para ele mesmo é identidade

    for i in range(len(pcds_down) - 1):
        source_idx = i + 1
        target_idx = 0 # Alinha sempre ao primeiro fragmento (registro em estrela)
        
        print(f"   Alinhando fragmento {source_idx+1} (source) ao fragmento {target_idx+1} (target)...")
        
        # Registro Global
        result_global = execute_global_registration(pcds_down[source_idx], pcds_down[target_idx], 
                                                    pcds_fpfh[source_idx], pcds_fpfh[target_idx], VOXEL_SIZE_GLOBAL_REG)
        
        # Refinamento com ICP
        result_fine = refine_registration(pcds_down[source_idx], pcds_down[target_idx], 
                                          result_global.transformation, VOXEL_SIZE_GLOBAL_REG)
        
        transformations.append(result_fine.transformation) 

    # Combinar todas as nuvens de pontos na resolução original usando as transformações
    print("\n   Combinando todas as nuvens de pontos alinhadas em uma única nuvem (resolução original)...")
    pcd_combined_full_res = o3d.geometry.PointCloud()
    for i, pcd_orig in enumerate(pcds_original_res):
        pcd_transformed = copy.deepcopy(pcd_orig).transform(transformations[i])
        pcd_combined_full_res += pcd_transformed
    
    # Aplicar limpeza e downsampling final na nuvem combinada completa
    pcd_combined_full_res_cleaned = pcd_combined_full_res.voxel_down_sample(voxel_size=VOXEL_SIZE_FINAL_PCD)
    cl, ind = pcd_combined_full_res_cleaned.remove_statistical_outlier(nb_neighbors=SOR_NB_NEIGHBORS, std_ratio=SOR_STD_RATIO)
    pcd_combined_full_res_cleaned = cl
    
    print(f"   Nuvem de pontos combinada e limpa: {len(pcd_combined_full_res_cleaned.points)} pontos.")
    o3d.io.write_point_cloud("01_ambiente_completo_limpo.ply", pcd_combined_full_res_cleaned)
    print("   Salvo: 01_ambiente_completo_limpo.ply")


    print("\n--- PASSO 3: Detecção e Remoção Iterativa de Planos (Chão, Paredes, Teto) ---")
    pcd_remaining_after_planes = copy.deepcopy(pcd_combined_full_res_cleaned)
    planes_detected_pcds = []
    
    for i in range(MAX_PLANES_TO_REMOVE):
        print(f"   Tentativa {i+1}/{MAX_PLANES_TO_REMOVE}: Detectando plano...")
        model, inliers = pcd_remaining_after_planes.segment_plane(distance_threshold=RANSAC_DISTANCE_THRESHOLD,
                                                                 ransac_n=RANSAC_N,
                                                                 num_iterations=RANSAC_NUM_ITERATIONS)
        if len(inliers) < 100:
            print(f"     Plano muito pequeno ou nenhum plano significativo encontrado. Parando detecção de planos.")
            break

        plane_pcd = pcd_remaining_after_planes.select_by_index(inliers)
        plane_pcd.paint_uniform_color([np.random.rand(), np.random.rand(), np.random.rand()])
        planes_detected_pcds.append(plane_pcd)
        
        pcd_remaining_after_planes = pcd_remaining_after_planes.select_by_index(inliers, invert=True)
        print(f"     Plano detectado com {len(inliers)} pontos. Pontos restantes: {len(pcd_remaining_after_planes.points)}")
    
    o3d.io.write_point_cloud("02_ambiente_sem_planos.ply", pcd_remaining_after_planes)
    print("   Salvo: 02_ambiente_sem_planos.ply (contém os objetos)")


    print("\n--- PASSO 4: Clustering e Reconstrução de Malha para Objetos ---")
    output_objects_folder_ply = "objetos_isolados_ply"
    output_objects_folder_obj = "objetos_modelados_obj"
    os.makedirs(output_objects_folder_ply, exist_ok=True)
    os.makedirs(output_objects_folder_obj, exist_ok=True)

    isolated_objects_count = 0
    
    if not pcd_remaining_after_planes.has_points():
        print("   Nenhum ponto restante para clustering após remoção de planos. Pulando clusterização.")
        return pcd_combined_full_res_cleaned, planes_detected_pcds, []

    labels = np.array(pcd_remaining_after_planes.cluster_dbscan(eps=DBSCAN_EPS, min_points=DBSCAN_MIN_POINTS, print_progress=True))
    max_label = labels.max()
    print(f"   Número total de clusters detectados: {max_label + 1} (incluindo ruído -1).")

    objects_pcds = []
    if max_label >= 0:
        for i in range(max_label + 1):
            if i == -1: # Ignorar ruído
                continue
            cluster_indices = np.where(labels == i)[0]
            if len(cluster_indices) >= DBSCAN_MIN_POINTS:
                cluster_pcd = pcd_remaining_after_planes.select_by_index(cluster_indices)
                objects_pcds.append(cluster_pcd)

                # Salvar o cluster PLY
                output_ply_path = os.path.join(output_objects_folder_ply, f"objeto_cluster_{i}.ply")
                o3d.io.write_point_cloud(output_ply_path, cluster_pcd)
                print(f"     Cluster {i} salvo em: {output_ply_path} ({len(cluster_pcd.points)} pontos)")

                # Reconstruir Malha do Objeto
                print(f"     Reconstruindo malha para Objeto {i}...")
                # Opcional: Downsample antes da reconstrução para otimização
                obj_pcd_for_reconstruction = cluster_pcd.voxel_down_sample(voxel_size=VOXEL_SIZE_OBJ_RECONSTRUCTION)
                
                if len(obj_pcd_for_reconstruction.points) < 100: # Evitar reconstruir objetos minúsculos
                    print(f"       Objeto {i} muito pequeno ({len(obj_pcd_for_reconstruction.points)} pts) para reconstrução. Pulando.")
                    continue

                obj_pcd_for_reconstruction.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
                    radius=NORMAL_SEARCH_RADIUS, max_nn=NORMAL_MAX_NEIGHBORS))
                obj_pcd_for_reconstruction.orient_normals_to_align_with_direction(orientation_reference=np.array([0., 0., 1.]))
                
                mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(obj_pcd_for_reconstruction, depth=POISSON_DEPTH)
                
                # Cortar mesh pela bounding box original para limpar artefatos
                bbox = obj_pcd_for_reconstruction.get_axis_aligned_bounding_box()
                mesh = mesh.crop(bbox)
                
                # Suavização opcional
                mesh = mesh.filter_smooth_laplacian(number_of_iterations=3) # Suaviza levemente
                mesh.compute_vertex_normals()

                output_obj_path = os.path.join(output_objects_folder_obj, f"objeto_modelado_{i}.obj")
                o3d.io.write_triangle_mesh(output_obj_path, mesh)
                print(f"     Malha do Objeto {i} salva em: {output_obj_path}")
                isolated_objects_count += 1
    else:
        print("   Nenhum cluster válido (com pontos suficientes) foi detectado.")

    print(f"\nTotal de {isolated_objects_count} objetos isolados e modelados salvos.")
    return pcd_combined_full_res_cleaned, planes_detected_pcds, objects_pcds


# --- EXECUÇÃO DO PIPELINE ---
if __name__ == "__main__":
    print("\n--- INICIANDO O ROBÔ DE PROCESSAMENTO LIDAR ---")

    # 0. Gerar fragments simulados (para teste)
    # No cenário real, input_fragments_paths seria uma lista de arquivos .ply que você CAPTUROU
    print("\n0. Gerando fragments simulados para o teste do pipeline...")
    if not os.path.exists(BASE_FRAGMENT_FILE):
        print(f"ERRO: O arquivo base '{BASE_FRAGMENT_FILE}' não foi encontrado.")
        print("Por favor, baixe 'fragment.ply' do Open3D e coloque-o na mesma pasta do script.")
        exit()
    
    base_pcd_for_sim = o3d.io.read_point_cloud(BASE_FRAGMENT_FILE)

    # Fragmento 1 (Referência)
    pcd_sim1 = copy.deepcopy(base_pcd_for_sim).translate([-0.5, 0, 0])
    # Fragmento 2 (Deslocado)
    pcd_sim2 = copy.deepcopy(base_pcd_for_sim)
    pcd_sim2.translate([0.5, 0, 0.1])
    pcd_sim2.rotate(pcd_sim2.get_rotation_matrix_from_xyz((0.1, 0, 0)))
    # Fragmento 3 (Mais Deslocado)
    pcd_sim3 = copy.deepcopy(base_pcd_for_sim)
    pcd_sim3.translate([0, 0.5, -0.1])
    pcd_sim3.rotate(pcd_sim3.get_rotation_matrix_from_xyz((0, 0.2, 0)))

    # Salvar temporariamente para simular arquivos de entrada
    temp_dir = "temp_fragments"
    os.makedirs(temp_dir, exist_ok=True)
    o3d.io.write_point_cloud(os.path.join(temp_dir, "sim_fragment_1.ply"), pcd_sim1)
    o3d.io.write_point_cloud(os.path.join(temp_dir, "sim_fragment_2.ply"), pcd_sim2)
    o3d.io.write_point_cloud(os.path.join(temp_dir, "sim_fragment_3.ply"), pcd_sim3)

    input_fragments_paths = [
        os.path.join(temp_dir, "sim_fragment_1.ply"),
        os.path.join(temp_dir, "sim_fragment_2.ply"),
        os.path.join(temp_dir, "sim_fragment_3.ply")
    ]
    print(f"   {len(input_fragments_paths)} fragments simulados salvos em '{temp_dir}' para o pipeline.")


    # Executar o pipeline completo
    final_pcd_ambiente, detected_planes, isolated_objects = run_full_pipeline(input_fragments_paths)

    # --- VISUALIZAÇÃO FINAL DE TUDO ---
    if final_pcd_ambiente:
        print("\n--- VISUALIZAÇÃO FINAL: Ambiente Completo e Objetos Isolados (se houver) ---")
        
        # Pinta o ambiente de cinza claro
        final_pcd_ambiente.paint_uniform_color([0.7, 0.7, 0.7]) 
        
        # Pinta os objetos isolados com cores diferentes para visualização final
        viz_objects = []
        if isolated_objects:
            colors_map_obj = plt.get_cmap("tab10") # Novo mapa de cores para objetos
            for i, obj_pcd in enumerate(isolated_objects):
                if obj_pcd.has_points():
                    obj_pcd_colored = copy.deepcopy(obj_pcd)
                    obj_pcd_colored.paint_uniform_color(colors_map_obj(i % 10)[:3]) # Pinta com cor do cmap
                    viz_objects.append(obj_pcd_colored)

        print("   Visualizando o ambiente completo e os objetos separados. Feche a janela para finalizar.")
        o3d.visualization.draw_geometries([final_pcd_ambiente] + viz_objects,
                                          window_name="Pipeline Completo: Ambiente e Objetos",
                                          width=1200, height=800, left=50, top=50)
        print("   Visualização final encerrada.")
    else:
        print("\n   Pipeline não produziu resultados visuais finais.")

    print("\n--- ROBÔ DE PROCESSAMENTO LIDAR CONCLUÍDO! ---")
