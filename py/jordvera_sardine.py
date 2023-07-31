import tolvera as tol
from iml import IML
import torch
x=1920
y=1080
n=4096
species=1
fps=12
camera=0
camera_substeps=1 # cv

'''''''''''''''''''''''''''''''''''''''''''''
Tolvera
'''''''''''''''''''''''''''''''''''''''''''''
tol.init(x=x, y=y, n=n, species=species, fps=fps)
particles = tol.Particles(x, y, n, species, wall_margin=0)
pixels    = tol.Pixels(x, y)
boids     = tol.vera.Boids(x, y, species)
physarum  = tol.vera.Physarum(x, y, species)

'''''''''''''''''''''''''''''''''''''''''''''
CV
'''''''''''''''''''''''''''''''''''''''''''''
cv = tol.CV(x, y, fps, camera_substeps, device=camera)
factor = 4
_img = cv.cv2_camera_read()
img = cv.cv2_pyr_down(cv.cv2_bgr_to_gray(_img), factor)
def process_cv(img, _img):
    # nonlocal cv
    cv.i += 1
    if cv.i % cv.camera_substeps == 0:
        frame       = cv.cv2_camera_read()
        grayscale   = cv.cv2_bgr_to_gray(frame)
        inverted    = cv.cv2_invert(grayscale)
        downsampled = cv.cv2_pyr_down(inverted, factor)
        blurred     = cv.cv2_gaussian_blur(downsampled, (1, 25), 0)
        img         = blurred
        upsampled   = cv.cv2_pyr_up(blurred, factor)
        _img        = cv.img2px_rgba(upsampled) # render
        cv.frame2px_g(upsampled) # render_physarum
    return img, _img

img, _img = process_cv(img, _img)
img_flat = img.flatten()

'''''''''''''''''''''''''''''''''''''''''''''
IML
'''''''''''''''''''''''''''''''''''''''''''''
iml_source_size = img_flat.shape[0]
iml_target_size = 16
iml_source = torch.zeros(iml_source_size)
iml_target = torch.zeros(iml_target_size)
iml_target_norm = torch.zeros(iml_target_size)
iml = IML(iml_source_size)
def iml_randomise():
    while(len(iml.pairs) < 32):
        source = torch.rand(iml_source_size)#/(ctrl.abs()/2+1)
        target = iml_target + torch.randn(iml_target_size)*2#/(z.abs()/2+1)
        iml.add(source, target)

iml_randomise()

def iml_print(tensor):
    print(" ".join([f"{i:.2f}" for i in tensor.tolist()]))

def iml_norm(tensor, min=-1, max=1):
    return 2 * (tensor - min) / (max - min) - 1

@swim
def render_cv(p=0.5, i=0):
    global img, _img
    img, _img = process_cv(img, _img)
    pixels.clear()
    pixels.px.rgba = cv.px_rgba
    tol.show(pixels)
    again(render_cv, p=1/64, i=i+1)

@swim
def iml_update(p=0.5, i=0):
    global img
    iml_source = torch.from_numpy(img.flatten())
    iml_target[:] = torch.from_numpy(iml.map(iml_source, k=5))#.float()
    iml_target_norm = iml_norm(iml_target, -6, 6)
    iml_print(iml_target_norm)
    again(iml_update, p=1/16, i=i+1)

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
    # iml_target_norm
    D("hh", d=1, rate=0.125, i=i, gain=gain())
    D("bd", d=8, rate=1, i=i, gain="1")
    again(control_superdirt, p=0.25, i=i+1)

@swim
def control_tolvera(p=4, i=0):
    pixels.evaporate[None] = P('0.99 0.1', i)
    particles.wall_margin[None] = P('100 0', i)
    again(control_tolvera, p=1/2, i=i+1)

silence()

