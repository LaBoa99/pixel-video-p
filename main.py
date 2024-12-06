import argparse
import os

import numpy as np
import pygame
from moviepy.editor import VideoFileClip


class Pixel(pygame.sprite.Sprite):
    def __init__(self, w, h, x, y, *groups) -> None:
        super().__init__(*groups)
        self.image = pygame.Surface((w, h))
        self.rect = self.image.get_rect()
        self.rect.topleft = (x, y)
        self.setColor((255, 255, 255))
    
    def setColor(self, color):
        self.color = color
        self.image.fill(self.color)

class PixelVideo:
    
    def __init__(self, cell_w, cell_h, rows, cols, loadNumpy = False) -> None:
        self.cell_w = cell_w
        self.cell_h = cell_h
        self.rows = rows
        self.cols = cols  
        self.loadNumpy = loadNumpy if loadNumpy else False
    
    def paths(self, filepath: str):
        scrip_dir = os.path.dirname(__file__)
        videopath = os.path.join(scrip_dir, "videos", filepath)
        filename = filepath.split(".")[0]
        return scrip_dir, videopath, filename
    
    def metadata(self, filepath: str):
        scrip_dir, videopath, filename = self.paths(filepath)
        with VideoFileClip(videopath) as clip:
            count_frames = int(clip.duration * clip.fps)
            return count_frames, clip.fps
    
    def __load(self, filepath: str):
        scrip_dir, videopath, filename = self.paths(filepath)
        data = np.load(os.path.join(scrip_dir, "outputs", filepath))
        return data
    
    # Retorna frames, count_frames, rows, cols, 
    def load(self, inputpath, outputpath=""):
        frames = self.__load(inputpath)
        return frames, frames.shape[0], frames.shape[1], frames.shape[2] 
    
    def analyze(self, filepath: str):
        scrip_dir, videopath, filename = self.paths(filepath)
        with VideoFileClip(videopath) as clip:
            frames = clip.iter_frames()
            count_frames = int(clip.duration * clip.fps)
            frames_raw = np.empty((count_frames, self.rows, self.cols, 3), dtype=np.uint8)
            
            print("Empezando analisis")
            for i, frame in enumerate(frames):
                print(f"{i}/{count_frames}")
                for row in range(rows):
                    start_row, end_row = (self.cell_h) * row, self.cell_h * (row + 1)
                    for col in range(cols):
                        try:
                            start_col, end_col = self.cell_w * col, self.cell_w * (col + 1)
                            roi = frame[start_row:end_row, start_col:end_col]
                            # Convierte la ROI a una dimensión para calcular la moda
                            flattened_roi = roi.reshape(-1, 3)

                            # Encuentra la moda en la ROI considerándola como una entidad única
                            mode_pixel = tuple(np.median(flattened_roi, axis=0).astype(int))
                            if len(mode_pixel) == 3:
                                frames_raw[i, row, col, 0] = mode_pixel[0]
                                frames_raw[i, row, col, 1] = mode_pixel[1]
                                frames_raw[i, row, col, 2] = mode_pixel[2]
                        except IndexError:
                            continue
            print("Terminado")
            np.save(os.path.join(scrip_dir, "outputs", f"output_{filename}"), frames_raw)
            return frames_raw
        
# Argumentos
parser = argparse.ArgumentParser(description='Ejemplo de script con argumento.')
parser.add_argument('-l', '--load', action='store_true', help='Especifica si se debe cargar algo.')
parser.add_argument('-i', '--fileinput', type=str, help='Especifica si se debe cargar algo.')
parser.add_argument('-o', '--fileoutput', type=str, help='Especifica si se debe cargar algo.')
parser.add_argument('-f', '--fps', type=int, help='Especifica si se debe cargar algo.')
parser.add_argument('-r', '--rows', type=int, help='Especifica si se debe cargar algo.')
parser.add_argument('-c', '--cols', type=int, help='Especifica si se debe cargar algo.')
args = parser.parse_args()


       
# Configuración
W, H = (480, 480)
rows = args.rows if args.rows else 64
cols = args.cols if args.cols else 64
cell_w = W // cols
cell_h = H // rows
loadNumpy = args.load
input_path = args.fileinput
data_name = args.fileoutput
fps = args.fps if args.fps else 30

pixelVideo = PixelVideo(cell_w, cell_h, rows, cols)

# Cargar o analizar video
if loadNumpy:
    frames_raw, count_frames, rows, cols = pixelVideo.load(input_path)
else:
    frames_raw = pixelVideo.analyze(input_path)
    count_frames, fps = pixelVideo.metadata(input_path)

# =======================================================
# =======================================================
# Inicio de pygame
pygame.init()

# Crear pixeles
pixels = pygame.sprite.Group()
surface = np.empty((rows, cols), dtype=object)
for row in range(rows):
    for col in range(cols):
        pixel = Pixel(cell_w, cell_h, cell_w * col, cell_h * row, pixels)
        surface[row, col] = pixel
        
# Configuracion de pantalla y reloj
pantalla = pygame.display.set_mode((W, H))
clock = pygame.time.Clock()
current_time = pygame.time.get_ticks()
previous_time = current_time

# Bucle Principall
current_frame = 0
while True:
    # Eventos
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            break
    pantalla.fill((0, 0, 0))
    
    deltatime = (current_time - previous_time) / 1000.0  # Tiempo en segundos
    previous_time = deltatime
    # update
    frame_raw = frames_raw[current_frame]
    for row in range(frame_raw.shape[0]):
        for col in range(frame_raw.shape[1]):
            rbg_values = frame_raw[row, col]
            surface[row, col].setColor(tuple(rbg_values))
            
    pixels.update(deltatime)
    # Draw
    pixels.draw(pantalla)
    # Final
    pygame.display.flip()
    clock.tick(fps)
    current_frame = (current_frame + 1) % count_frames