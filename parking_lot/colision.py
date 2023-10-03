import cv2
import numpy as np

# Função para verificar a colisão entre dois retângulos
def check_collision(rect1, rect2):
    # Extrair os pontos dos retângulos
    rect1_pts = np.array(rect1)
    rect2_pts = np.array(rect2)

    # Calcular os limites dos retângulos
    rect1_x_min, rect1_y_min = np.min(rect1_pts, axis=0)
    rect1_x_max, rect1_y_max = np.max(rect1_pts, axis=0)
    
    rect2_x_min, rect2_y_min = np.min(rect2_pts, axis=0)
    rect2_x_max, rect2_y_max = np.max(rect2_pts, axis=0)

    # Verificar a colisão
    if (rect1_x_max < rect2_x_min or rect1_x_min > rect2_x_max or
        rect1_y_max < rect2_y_min or rect1_y_min > rect2_y_max):
        return False
    else:
        return True

# Definir os pontos dos retângulos
rect1 = [[532, 183], [524, 278], [590, 287], [594, 186]]
rect2 = [[542, 183], [524, 278], [600, 287], [594, 186]]
#rect1 = [[65, 343], [55, 445], [133, 440], [153, 342]]

# Criar uma imagem em branco para desenhar os retângulos
image = np.zeros((500, 800, 3), dtype=np.uint8)

# Desenhar os retângulos na imagem
cv2.rectangle(image, tuple(rect1[0]), tuple(rect1[2]), (0, 255, 0), -1)
cv2.rectangle(image, tuple(rect2[0]), tuple(rect2[2]), (0, 0, 255), -1)

# Verificar a colisão
collision = check_collision(rect1, rect2)

# Exibir a imagem com os retângulos
cv2.imshow("Rectangles", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Exibir o resultado da colisão
if collision:
    print("Houve colisão entre os retângulos.")
else:
    print("Não houve colisão entre os retângulos.")
