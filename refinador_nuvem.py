import open3d as o3d
import numpy as np
import os # Módulo para interagir com o sistema operacional (caminhos de arquivo)

# --- CONFIGURAÇÕES ---
# 1. Nome do arquivo de entrada (sua nuvem de pontos .ply)
# Certifique-se de que este arquivo esteja na mesma pasta do script,
# ou forneça o caminho completo para ele.
input_file_name = "caneca_generica.ply" 

# 2. Nomes dos arquivos de saída para salvar os resultados
output_file_cleaned = "nuvem_processada_limpa.ply"
output_file_downsampled = "nuvem_processada_limpa_e_reduzida.ply"

# 3. Parâmetros para o Filtro Estatístico (Statistical Outlier Removal - SOR)
# nb_neighbors: Quantos vizinhos considerar para calcular a distância média de um ponto.
#               Valores comuns: 10 a 30.
# std_ratio: Desvio padrão. Pontos cuja distância média é maior que (std_ratio * desvio_padrao_global)
#            serão considerados outliers. Valores comuns: 1.0 a 3.0.
#            Um valor menor remove mais pontos (é mais agressivo).
sor_nb_neighbors = 20
sor_std_ratio = 2.0

# 4. Parâmetros para o Downsampling (Voxel Grid)
# voxel_size: Tamanho do voxel (em unidades da sua nuvem de pontos, geralmente metros).
#             Um voxel de 0.02 (2 cm) significa que pontos dentro de um cubo de 2x2x2 cm
#             serão representados por um único ponto.
#             Um valor maior reduz mais pontos, mas pode perder detalhes finos.
voxel_grid_size = 0.02 # Exemplo: 2 centímetros

# --- INÍCIO DO SCRIPT ---
print("--- Iniciando o Processamento da Nuvem de Pontos ---")

# 0. Verificar a existência do arquivo de entrada
if not os.path.exists(input_file_name):
    print(f"ERRO: O arquivo '{input_file_name}' não foi encontrado.")
    print("Por favor, verifique se o nome do arquivo está correto e se ele está na mesma pasta do script,")
    print("ou forneça o caminho completo para o arquivo.")
    exit() # Sai do script se o arquivo não for encontrado

# 1. Carregar a Nuvem de Pontos
print(f"\n1. Carregando a nuvem de pontos do arquivo: {input_file_name}")
pcd_original = o3d.io.read_point_cloud(input_file_name)
print(f"   Nuvem de pontos original: {len(pcd_original.points)} pontos.")

# 2. Visualizar a Nuvem de Pontos Original
print("\n2. Visualizando a nuvem de pontos ORIGINAL. Feche a janela para continuar...")
o3d.visualization.draw_geometries([pcd_original],
                                  window_name="Nuvem Original",
                                  width=800, height=600,
                                  left=50, top=50)
print("   Visualização da nuvem original encerrada.")

# 3. Remover Ruído (Filtro Estatístico - SOR)
print(f"\n3. Aplicando filtro de ruído (Statistical Outlier Removal) com:")
print(f"   - Vizinhos (nb_neighbors): {sor_nb_neighbors}")
print(f"   - Proporção de Desvio Padrão (std_ratio): {sor_std_ratio}")

# cl: 'cleaned' (nuvem de pontos limpa), ind: índices dos pontos que foram mantidos
cl, ind = pcd_original.remove_statistical_outlier(nb_neighbors=sor_nb_neighbors, std_ratio=sor_std_ratio)
pcd_cleaned = cl # Renomeando para clareza

print(f"   Nuvem de pontos após filtro de ruído: {len(pcd_cleaned.points)} pontos.")
# Opcional: Para ver os pontos removidos, descomente as linhas abaixo
# outliers = pcd_original.select_by_index(ind, invert=True)
# outliers.paint_uniform_color([1, 0, 0]) # Pinta os outliers de vermelho
# o3d.visualization.draw_geometries([pcd_cleaned, outliers], window_name="Nuvem Limpa + Outliers (Vermelho)")


# 4. Reduzir a Densidade (Downsampling - Voxel Grid)
print(f"\n4. Aplicando downsampling (Voxel Grid) com tamanho de voxel: {voxel_grid_size} metros")
pcd_downsampled = pcd_cleaned.voxel_down_sample(voxel_size=voxel_grid_size)

print(f"   Nuvem de pontos após downsampling: {len(pcd_downsampled.points)} pontos.")
print(f"   Redução total de pontos: {len(pcd_original.points) - len(pcd_downsampled.points)} pontos.")
print(f"   Percentual de redução: {((len(pcd_original.points) - len(pcd_downsampled.points)) / len(pcd_original.points)) * 100:.2f}%")

# 5. Visualizar a Nuvem de Pontos Refinada (Limpa e Reduzida)
print("\n5. Visualizando a nuvem de pontos REFINADA. Feche a janela para continuar...")
o3d.visualization.draw_geometries([pcd_downsampled],
                                  window_name="Nuvem Refinada",
                                  width=800, height=600,
                                  left=900, top=50) # Posição para não sobrepor a primeira janela
print("   Visualização da nuvem refinada encerrada.")

# 6. Salvar os Resultados Processados
print(f"\n6. Salvando a nuvem de pontos limpa em: {output_file_cleaned}")
o3d.io.write_point_cloud(output_file_cleaned, pcd_cleaned)

print(f"   Salvando a nuvem de pontos limpa e downsampled em: {output_file_downsampled}")
o3d.io.write_point_cloud(output_file_downsampled, pcd_downsampled)

print("\n--- Processamento da Nuvem de Pontos Concluído! ---")
print("Você pode agora inspecionar os arquivos gerados em sua pasta.")
