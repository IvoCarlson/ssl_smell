import os
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
from scipy.signal import convolve2d

def rotate_img(img, rot):
    if rot == 0:  # 0 degrees rotation
        return img
    elif rot == 90:  # 90 degrees rotation
#        return np.flipud(np.transpose(img, (1, 0, 2)))
        return img.rotate(90)
    elif rot == 180:  # 90 degrees rotation
#        return np.fliplr(np.flipud(img))
        return img.rotate(180)
    elif rot == 270:  # 270 degrees rotation / or -90
#        return np.transpose(np.flipud(img), (1, 0, 2))
        return img.rotate(270)
    elif rot == 120:
#        return ndimage.rotate(img, 120, reshape=False)
        return img.rotate(120)
    elif rot == 240:
#        return ndimage.rotate(img, 240, reshape=False)
        return img.rotate(240)
    else:
        raise ValueError('rotation should be 0, 90, 120, 180, 240 or 270 degrees')

def relative_patch(img):
    """
    Divide a imagem em uma grade 3x3, seleciona dois patches aleatórios e retorna
    os patches com suas posições relativas.

    Args:
        img (PIL.Image.Image): A imagem a ser dividida em patches.

    Returns:
        tuple: (patch1, patch2, position1, position2)
            - patch1, patch2: as duas imagens de patch selecionadas.
            - position1, position2: as posições relativas dos patches selecionados.
    """

    # Converte a imagem para numpy array (se necessário)
    img_array = np.array(img)

    # Obtem as dimensões da imagem
    h, w, _ = img_array.shape
    grade=5 # 2 3 4 5 6 7
    # Calculando as dimensões dos patches (assumindo que a imagem é divisível por 3)
    patch_height = h // grade
    patch_width = w // grade

    # Lista de posições de todos os patches (3x3)
    patches = []

    for row in range(grade):
        for col in range(grade):
            # Corta a imagem para pegar o patch
            patch = img_array[row * patch_height : (row + 1) * patch_height,
                              col * patch_width : (col + 1) * patch_width]
            patches.append((patch, (row, col)))

    # Seleciona dois patches aleatórios
    patch1, position1 = patches[np.random.choice(len(patches))]
    patch2, position2 = patches[np.random.choice(len(patches))]

    # Converte os patches de volta para imagens PIL
    patch1 = Image.fromarray(patch1)
    patch2 = Image.fromarray(patch2)

    return patch1, patch2, position1, position2

def add_gaussian_noise(img, mean=0, std=1):
    """ Adiciona ruído Gaussiano à imagem """
    img_array = np.array(img)
    noise = np.random.normal(mean, std, img_array.shape)
    noisy_img_array = img_array + noise
    noisy_img_array = np.clip(noisy_img_array, 0, 255)
    noisy_img = Image.fromarray(noisy_img_array.astype(np.uint8))
    return noisy_img

def add_salt_and_pepper_noise(img, amount=0.02):
    """ Adiciona ruído Sal e Pimenta à imagem """
    img_array = np.array(img)
    row, col, _ = img_array.shape
    s_vs_p = 0.5
    num_salt = int(amount * row * col * s_vs_p)
    num_pepper = int(amount * row * col * (1.0 - s_vs_p))

    # Adiciona os pixels de sal
    salt_coords = [np.random.randint(0, i-1, num_salt) for i in img_array.shape]
    img_array[salt_coords[0], salt_coords[1], :] = 1

    # Adiciona os pixels de pimenta
    pepper_coords = [np.random.randint(0, i-1, num_pepper) for i in img_array.shape]
    img_array[pepper_coords[0], pepper_coords[1], :] = 0

    noisy_img = Image.fromarray(img_array)
    return noisy_img

def add_poisson_noise(img, scale=1.0):
    """Adiciona ruído de Poisson à imagem com um fator de escala para controlar a intensidade."""
    img_array = np.array(img, dtype=np.float32)  # Converte para float para evitar overflow
    img_array = img_array * scale  # Aplica o fator de escala (controle da intensidade)

    # Aplica a distribuição de Poisson
    noisy_img_array = np.random.poisson(img_array)

    # Clipping para garantir que os valores fiquem dentro do intervalo 0-255
    noisy_img_array = np.clip(noisy_img_array, 0, 255)

    noisy_img = Image.fromarray(noisy_img_array.astype(np.uint8))  # Converte de volta para imagem
    return noisy_img

def add_speckle_noise(img, mean=0, std=0.1):
    """ Adiciona ruído Speckle à imagem """
    img_array = np.array(img)
    noise = np.random.normal(mean, std, img_array.shape)
    noisy_img_array = img_array + img_array * noise
    noisy_img_array = np.clip(noisy_img_array, 0, 255)
    noisy_img = Image.fromarray(noisy_img_array.astype(np.uint8))
    return noisy_img

def add_uniform_noise(img, low=0, high=0.1):
    """ Adiciona ruído uniforme à imagem """
    img_array = np.array(img)
    noise = np.random.uniform(low, high, img_array.shape)
    noisy_img_array = img_array + noise * 255
    noisy_img_array = np.clip(noisy_img_array, 0, 255)
    noisy_img = Image.fromarray(noisy_img_array.astype(np.uint8))
    return noisy_img

def get_noise_intensity(noise_type, level):
    noise_intensities = {
        'gaussian': [2, 5, 10, 20, 30, 40, 50],            # Intensidade para Gaussian
        'salt_and_pepper': [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3],  # Intensidade para Salt and Pepper
        'poisson': [0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 4.0],  # Intensidade para Poisson
        'speckle': [0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5],   # Intensidade para Speckle
        'uniform': [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]             # Intensidade para Uniform
    }
    if noise_type not in noise_intensities:
        raise ValueError(f"Unknown noise type: {noise_type}")
    if not (0 <= level <= 6):
        raise ValueError("Level must be between 0 and 6")
    return noise_intensities[noise_type][level]

def add_random_noise(img, noise_type=None, level=5):
    noise_types = ['gaussian', 'salt_and_pepper', 'poisson', 'speckle', 'uniform']

    if noise_type is None:
        # Escolhe aleatoriamente um tipo de ruído se nenhum for especificado
        noise_type = np.random.choice(noise_types)

    if noise_type == 'gaussian':
        noisy_img = add_gaussian_noise(img, mean=0, std=get_noise_intensity('gaussian',level))
        noise_label = 0
    elif noise_type == 'salt_and_pepper':
        noisy_img = add_salt_and_pepper_noise(img, amount=get_noise_intensity('salt_and_pepper',level))
        noise_label = 1
    elif noise_type == 'poisson':
        noisy_img = add_poisson_noise(img, scale=get_noise_intensity('poisson',level))
        noise_label = 2
    elif noise_type == 'speckle':
        noisy_img =  add_speckle_noise(img, mean=0, std=get_noise_intensity('speckle',level))
        noise_label = 3
    elif noise_type == 'uniform':
        noisy_img = add_uniform_noise(img, low=0, high=get_noise_intensity('uniform',level))
        noise_label = 4
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")

    return noisy_img, noise_label

   # Funções de desfoque
def add_gaussian_blur(img, radius=5):
    """ Adiciona desfoque Gaussiano (Gaussian Blur) à imagem """
    img_blurred = img.filter(ImageFilter.GaussianBlur(radius))
    return img_blurred

def add_radial_blur(img, radius=5):
    """ Adiciona desfoque Radial à imagem (simulado) """
    # O Pillow não tem suporte direto para Radial Blur, mas podemos usar uma técnica de distorção simples
    # para criar um efeito similar.
    return img.filter(ImageFilter.GaussianBlur(radius))  # Pode ser customizado mais tarde para distorcer radialmente

def add_motion_blur(img, radius):
# Garantir que o raio seja ímpar
    if radius % 2 == 0:
        radius += 1  # Faz o raio ímpar, se necessário

    # Criar o kernel de movimento horizontal
    kernel = np.zeros((radius, radius))
    kernel[int((radius - 1) / 2), :] = np.ones(radius)  # Movimento horizontal

    # Normalizar o kernel
    kernel = kernel / np.sum(kernel)  # Normalizar para evitar mudanças drásticas na imagem

    # Converter a imagem para RGB, caso não esteja nesse formato
    img_rgb = img.convert('RGB')

    # Converter a imagem para array NumPy
    img_array = np.array(img_rgb)

    # Aplica a convolução em cada canal de cor (R, G, B)
    r, g, b = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]

    r_blurred = convolve2d(r, kernel, mode='same', boundary='wrap')
    g_blurred = convolve2d(g, kernel, mode='same', boundary='wrap')
    b_blurred = convolve2d(b, kernel, mode='same', boundary='wrap')

    # Combina os canais de volta em uma imagem
    img_blurred = np.stack((r_blurred, g_blurred, b_blurred), axis=-1)

    # Garantir que os valores da imagem estão no intervalo [0, 255]
    img_blurred = np.clip(img_blurred, 0, 255).astype(np.uint8)

    # Converter o array de volta para imagem
    img_result = Image.fromarray(img_blurred)

    return img_result

def add_box_blur(img, radius=5):
    """ Adiciona desfoque Box (Box Blur) à imagem """
    return img.filter(ImageFilter.BoxBlur(radius))

def add_surface_blur(img, radius=5, threshold=20):
    """ Adiciona desfoque de superfície (Surface Blur) à imagem """
    # Pillow não tem suporte direto para Surface Blur, então vamos simular
    return img.filter(ImageFilter.BoxBlur(radius))  # Isso pode ser melhorado, mas para simplificação vamos usar BoxBlur

# Função para aplicar um desfoque aleatório
def add_random_blur(img, blur_type=None):
    blur_types = ['gaussian', 'radial', 'motion', 'box', 'surface']

    radius = 17 # 3 5 7 9 11 13 15 19 17 21

    if blur_type is None:
        # Escolhe aleatoriamente um tipo de desfoque se nenhum for especificado
        blur_type = np.random.choice(blur_types)

    if blur_type == 'gaussian':
        blurred_img = add_gaussian_blur(img, radius)
        blur_label = 0
    elif blur_type == 'radial':
        blurred_img = add_radial_blur(img, radius)
        blur_label = 1
    elif blur_type == 'motion':
        blurred_img = add_motion_blur(img, radius)
        blur_label = 2
    elif blur_type == 'box':
        blurred_img = add_box_blur(img, radius)
        blur_label = 3
    elif blur_type == 'surface':
        blurred_img = add_surface_blur(img, radius)
        blur_label = 4
    else:
        raise ValueError(f"Tipo de desfoque desconhecido: {blur_type}")


    return blurred_img, blur_label
