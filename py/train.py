import numpy as np
import cv2

w,h = 480, 270
t_w, t_h = w//4, h//5

lin_grad_x = np.linspace(1, 0, w)
lin_grad_y = np.linspace(1, 0, h)[:,None]

sources = [np.full((h, w), 1.0),
          np.full((h, w), 0.0),
          np.tile(lin_grad_x, (h, 1)),
          np.tile(lin_grad_y, (1, w)),
          np.tile(lin_grad_x[::-1], (h, 1)),
          np.tile(lin_grad_y[::-1], (1, w)),
          np.dot(lin_grad_y, lin_grad_x[None,:]),
          np.dot(lin_grad_y, lin_grad_x[None,:])[:,::-1],
          np.dot(lin_grad_y[::-1], lin_grad_x[None,:]),
          np.dot(lin_grad_y[::-1], lin_grad_x[None,:])[:,::-1]]

targets = [np.full(16, -1),
           np.full(16,  1)]

def tile(image, w, h, t_w, t_h):
    return image.reshape(h//t_h, t_h, w//t_w, t_w)

tiles = tile(sources[2], w, h, t_w, t_h)
avg_values = np.mean(tiles, axis=(1, 3))
print(avg_values.flatten(), avg_values.flatten().shape)

def save_img(image, name="img.png"):
    cv2.imwrite(name, image*255)

def save_tile(tile, name="tile.png"):
    tile = (tile * 255).astype(np.uint8)
    save_img(tile, name)

# [save_img(image, f"img{i}.png") for i, image in enumerate(sources)]



