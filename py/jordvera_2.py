import iml as iml_module
from iml import IML
from iipyper import OSC, run, Updater, OSCSendUpdater, OSCMap, Audio
import numpy as np
import tolvera as tol
import torch
import taichi as ti

import sounddevice as sd
import sys

def main(x=1920, y=1080, n=4096, species=1, fps=120, # tolvera
         host="127.0.0.1", client="127.0.0.1", receive_port=5001, send_port=6010, # osc
         patch_type="Pd", patch_name="osc_controls", # patcher
         headless=False, headless_rate=1/60, gpu='vulkan', cpu=None, # taichi
         device=None, rave_path=None, checkpoint=None, # rave
         camera=0, camera_substeps=1, # cv
        ):
    
    '''''''''''''''''''''''''''''''''''''''''''''
    Tolvera
    '''''''''''''''''''''''''''''''''''''''''''''
    tol.init(gpu=gpu, cpu=cpu, headless=headless, headless_rate=headless_rate)
    particles = tol.Particles(x, y, n, species, wall_margin=0)
    pixels    = tol.Pixels(x, y)
    boids     = tol.vera.Boids(x, y, species)
    physarum  = tol.vera.Physarum(x, y, species)

    '''''''''''''''''''''''''''''''''''''''''''''
    CV
    '''''''''''''''''''''''''''''''''''''''''''''
    cv = tol.CV(x, y, fps, camera_substeps, device=camera)
    factor = 10
    _img = cv.cv2_camera_read()
    img = cv.cv2_pyr_down(cv.cv2_bgr_to_gray(_img), factor)
    def process_cv(img, _img):
        # nonlocal cv
        # cv.i += 1
        # if cv.i % cv.camera_substeps == 0:
        frame       = cv.cv2_camera_read()
        grayscale   = cv.cv2_bgr_to_gray(frame)
        inverted    = cv.cv2_invert(grayscale)
        # downsampled = cv.cv2_pyr_down(inverted, factor)
        downsampled = cv.cv2_resize(inverted, (4,4), 0)
        # blurred     = cv.cv2_gaussian_blur(downsampled, (51, 51), 0)
        img         = downsampled#blurred
        # upsampled   = cv.cv2_pyr_up(img, factor)
        upsampled   = cv.cv2_resize(img, interpolation=0)
        # _img        = cv.img2px_rgba(upsampled) # render
        _img = cv.frame2px_rgba(frame) # render
        # cv.frame2px_g(upsampled) # render_physarum

        # print(frame.shape)
        tiles = grayscale.reshape(y//4, 4, x//4, 4)
        avg = np.mean(tiles, axis=(1,3))

        print(img.shape , avg.shape)
        # print(f"{img}\n {avg}\n\n")
        return img, _img

    img, _img = process_cv(img, _img)
    img_flat = img.flatten()
    print(f"IML source shape: {img.shape} | {img_flat.shape}")
    
    '''''''''''''''''''''''''''''''''''''''''''''
    RAVE
    '''''''''''''''''''''''''''''''''''''''''''''
    def rave_setup():
        rave = torch.jit.load(rave_path)
        print(f"RAVE params: {rave.encode_params}")
        try:
            sr = rave.sr
        except Exception:
            sr = rave.sampling_rate
        def rave_callback(
                indata: np.ndarray, outdata: np.ndarray, #[frames x channels]
                frames: int, time, status):
            with torch.inference_mode():
                outdata[:,:] = rave.decode(iml_target[None,:,None])[:,0].T
        audio = Audio(
            device=device, dtype=np.float32,
            samplerate=sr, blocksize=rave.encode_params[-1], 
            callback=rave_callback)
        audio.stream.start()
    if rave_path is not None:
        rave_setup()

    '''''''''''''''''''''''''''''''''''''''''''''
    OSC Mapping
    '''''''''''''''''''''''''''''''''''''''''''''
    osc = OSC(host, receive_port, verbose=False, concurrent=True)
    client_name = "jordvera"
    osc.create_client(client_name, client, send_port)
    osc_map = OSCMap(osc, client_name, create_patch=False)

    '''
    Patcher → Python
    '''
    # io, update_rate = 'receive', 5

    # # Reset
    # @osc_map.add(io='receive', count=1)
    # def tolvera_reset():
    #     particles.reset()
    #     pixels.reset()
    #     boids.reset()

    # # Particles
    # @osc_map.add(active=(n,0,n), io=io, count=update_rate)
    # def particles_active(active: int):
    #     nonlocal particles
    #     particles.osc_set_active(active)
    # @osc_map.add(i=(0,0,species), active=(n,0,n), io=io, count=update_rate)
    # def particles_species_active(i: int, active: int):
    #     nonlocal particles
    #     particles.osc_set_species_active(i, active)
    # @osc_map.add(i=(0,0,species), r=(1,0,1), g=(1,0,1), b=(1,0,1), io=io, count=update_rate)
    # def species_color(i: int, r: float, g: float, b: float):
    #     nonlocal particles
    #     particles.osc_set_species_color(i, r, g, b)
    # @osc_map.add(i=(0,0,species), size=(3,0,10), io=io, count=update_rate)
    # def species_size(i: int, size: float):
    #     nonlocal particles
    #     particles.osc_set_species_size(i, size)
    # @osc_map.add(i=(0,0,species), speed=(1,0,10), max=(1,0,10), io=io, count=update_rate)
    # def species_speed(i: int, speed: float, max: float):
    #     nonlocal particles
    #     particles.osc_set_species_speed(i, speed, max)
    # @osc_map.add(margin=(50,0,200), turn=(0.8,0,5), io=io, count=update_rate)
    # def wall_repel(margin: float, turn: float):
    #     nonlocal particles
    #     particles.osc_set_wall_repel(margin, turn)

    # # Boids
    # @osc_map.add(a=(0,0,species), b=(0,0,species), 
    #              separate=(0.5,0,1), align=(0.5,0,1), cohere=(0.5,0,1), radius=(150,0,300), 
    #              io=io, count=update_rate)
    # def boids_rule(a: int, b: int, separate: float, align: float, cohere: float, radius: float):
    #     nonlocal boids
    #     boids.osc_set_rule(a, b, separate, align, cohere, radius)

    # IML
    # @osc_map.add("/tolvera/tgt")
    # def tgt_callback(msg):
    #     pass

    '''
    Python → Patcher
    '''
    io, update_rate = 'send', 25
    send_mode = 'broadcast' # | 'event'

    # IML
    # iml_target_counter = 0
    # @osc_map.add(io=io, count=1, send_mode='broadcast')
    # def ctrl() -> tuple[str, float]:
    #     nonlocal iml_target_norm, iml_target_size, iml_target_counter, iml_target_norm_prev
    #     key = "iml"+str(iml_target_counter)
    #     val = float(iml_target_norm[iml_target_counter])
    #     iml_target_counter += 1
    #     if iml_target_counter == iml_target_size:
    #         iml_target_counter = 0
    #     return [key, val]

    '''''''''''''''''''''''''''''''''''''''''''''
    Render
    '''''''''''''''''''''''''''''''''''''''''''''
    def render():
        nonlocal img, _img
        img, _img = process_cv(img, _img)
        osc_map()
        pixels.clear()
        pixels.px.rgba = cv.px_rgba

    tol.utils.render(render, pixels)

if __name__=='__main__':
    run(main)
