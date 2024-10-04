import cv2
import os

# Caminho para o vídeo
video_path = 'd:\Videos\pnct2.mp4'

# Diretório onde as imagens serão salvas
output_dir = 'd:\Videos\pnct2_video_imagens'
os.makedirs(output_dir, exist_ok=True)

# Captura o vídeo
cap = cv2.VideoCapture(video_path)

# Obtém a taxa de quadros do vídeo (frames per second - fps)
fps = cap.get(cv2.CAP_PROP_FPS)
# Calcula o intervalo de quadros (10 segundos)
intervalo_quadros = int(fps * 5)  # 10 segundos

frame_number = 0
extracted_frames = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Verifica se o quadro atual é um múltiplo do intervalo
    if frame_number % intervalo_quadros == 0:
        # Salva o quadro como imagem
        image_path = os.path.join(output_dir, f'frame_{extracted_frames:04d}.jpg')
        cv2.imwrite(image_path, frame)
        extracted_frames += 1

    frame_number += 1

cap.release()
cv2.destroyAllWindows()

print(f'Extração concluída! {extracted_frames} quadros foram salvos em {output_dir}')