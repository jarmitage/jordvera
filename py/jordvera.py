import iml as iml_module
from iml import IML
from iipyper import OSC, run, Updater, OSCSendUpdater, OSCMap, Audio
import numpy as np
import tolvera as tol
import torch

import sounddevice as sd
import sys

def main(x=1920, y=1080, n=4096, species=1, fps=120, # tolvera
         host="127.0.0.1", client="127.0.0.1", receive_port=5001, send_port=5000, # osc
         patch_type="Pd", patch_name="osc_controls", # patcher
         headless=False, headless_rate=1/60, gpu='vulkan', cpu=None, # taichi
         device=None, rave_path=None, checkpoint=None # rave
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
    cv = tol.CV(x, y, fps, colormode='g')
    factor = 4
    _img = cv.cv2_camera_read()
    img = cv.cv2_pyr_down(cv.cv2_bgr_to_gray(_img), factor)
    def process_cv(img, _img):
        # nonlocal cv
        cv.i += 1
        if cv.i % cv.camera_substeps == 0:
            frame       = cv.cv2_camera_read()
            grayscale   = cv.cv2_bgr_to_gray(frame)
            downsampled = cv.cv2_pyr_down(grayscale, factor)
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
    iml = IML(iml_source_size)
    def iml_randomise():
        while(len(iml.pairs) < 32):
            source = torch.rand(iml_source_size)#/(ctrl.abs()/2+1)
            target = iml_target + torch.randn(iml_target_size)*2#/(z.abs()/2+1)
            iml.add(source, target)
    iml_randomise()
    
    def iml_map():
        nonlocal img
        iml_source = torch.from_numpy(img.flatten())
        iml_target[:] = torch.from_numpy(iml.map(iml_source, k=5))#.float()
        print(f"target: {iml_target}")
    iml_update = Updater(iml_map, 24)

    # def iml_target_send():
    #     return iml_target.tolist()
    # osc_send_target = OSCSendUpdater(osc, "/tolvera/tgt", iml_target_send, fps)
    
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

    # '''''''''''''''''''''''''''''''''''''''''''''
    # OSC Mapping
    # '''''''''''''''''''''''''''''''''''''''''''''
    # osc = OSC(host, receive_port, verbose=True, concurrent=True)
    # client_name = "jordvera"
    # osc.create_client(client_name, client, send_port)
    # osc_map = OSCMap(osc, client_name, patch_type, patch_name)

    # '''
    # Patcher → Python
    # '''
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
    # io, update_rate = 'send', 25
    # send_mode = 'broadcast' # | 'event'

    # IML
    # @osc_map.add(io=io, count=update_rate, send_mode='broadcast')
    # def send_something() -> tuple[int, int]:
    #     pass

    '''''''''''''''''''''''''''''''''''''''''''''
    Render
    '''''''''''''''''''''''''''''''''''''''''''''
    def render():
        nonlocal img, _img
        img, _img = process_cv(img, _img)
        iml_update()
        pixels.clear()
        pixels.px.rgba = cv.px_rgba
    
    def render_physarum():
        nonlocal img, _img
        img, _img = process_cv(img, _img)
        iml_update()
        physarum.deposit_px(cv.px_g, 0.1)
        physarum(particles)
        pixels.set(physarum.trail.px)
        pixels.decay()

    # tol.utils.render(render, pixels)
    tol.utils.render(render_physarum, pixels)

if __name__=='__main__':
    run(main)
