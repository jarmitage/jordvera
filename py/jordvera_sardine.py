import tolvera as tol
from iml import IML
import torch

x,y,n,species,evaporate,fps=1920,1080,4096,1,0.95,120
tol.init(x=x,y=y,n=n,species=species,evaporate=evaporate,fps=fps)
particles = tol.Particles(x,y,n,species)
pixels = tol.Pixels(x,y, evaporate=evaporate, fps=fps)
# boids = tol.vera.Boids(x,y,species)
physarum = tol.vera.Physarum(x,y,species,evaporate)
cv = tol.CV(x,y,fps,colormode='g')


rave_params = 16
iml_src = torch.zeros(n*2)
iml_tgt = torch.zeros(rave_params)

img = cv.cv2_grayscale(cv.cv2_camera_read())
def process(img):
    global cv
    cv.i += 1
    if cv.i % cv.camera_substeps == 0:
        frame       = cv.cv2_camera_read()
        grayscale   = cv.cv2_grayscale(frame)
        downsampled = cv.cv2_pyr_down(grayscale, 6)
        blurred     = cv.cv2_gaussian_blur(downsampled, (125, 125), 0)
        img         = blurred
    return img

img = process(img)

flattened = torch.from_numpy(img.flatten())

print(flattened.shape)

@swim
def render_cv(p=0.5, i=0):
    global img
    img = process(img)
    cv.frame2px_g(cv.cv2_pyr_up(img, 6))
    # pixels.px.g = cv.px_g
    # pixels.g_to_rgba()
    # tol.show(pixels)
    again(render_cv, p=1/64, i=i+1)

@swim
def render_physarum_cv(p=0.5, i=0):
    physarum.deposit_px(cv(), 0.1)
    physarum(particles)
    pixels.set(physarum.trail.px)
    pixels.decay()
    tol.show(pixels)
    again(render_physarum_cv, p=1/64, i=i+1)

def gain(i=0):
    g = particles.osc_get_pos(i)[0]/1920
    print(g)
    return g

@swim
def control_superdirt(p=1, i=0):
    D("hh", d=1, rate=0.125, i=i, gain=gain())
    D("bd", d=8, rate=1, i=i, gain="1")
    again(control_superdirt, p=0.25, i=i+1)

@swim
def control_tolvera(p=4, i=0):
    pixels.evaporate[None] = P('0.99 0.1', i)
    particles.wall_margin[None] = P('100 0', i)
    again(control_tolvera, p=1/2, i=i+1)

silence()