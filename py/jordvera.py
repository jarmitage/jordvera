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
    factor = 2
    _img = cv.cv2_camera_read()
    img = cv.cv2_pyr_down(cv.cv2_bgr_to_gray(_img), factor)
    def process_cv(img, _img):
        # nonlocal cv
        # cv.i += 1
        # if cv.i % cv.camera_substeps == 0:
        frame       = cv.cv2_camera_read()
        grayscale   = cv.cv2_bgr_to_gray(frame)
        inverted    = cv.cv2_invert(grayscale)
        downsampled = cv.cv2_pyr_down(inverted, factor)
        blurred     = cv.cv2_gaussian_blur(downsampled, (51, 51), 0)
        img         = blurred
        upsampled   = cv.cv2_pyr_up(img, factor)
        _img        = cv.img2px_rgba(upsampled) # render
        cv.frame2px_g(upsampled) # render_physarum
        return img, _img

    img, _img = process_cv(img, _img)
    img_flat = img.flatten()
    print(f"IML source shape: {img.shape} | {img_flat.shape}")

    '''''''''''''''''''''''''''''''''''''''''''''
    IML
    '''''''''''''''''''''''''''''''''''''''''''''
    iml_source_size = img_flat.shape[0]
    iml_target_size = 20 # 16: rave latent vector size
    iml_source = torch.zeros(iml_source_size)
    iml_target = torch.zeros(iml_target_size)
    iml_target_norm = torch.zeros(iml_target_size)
    iml_target_norm_prev = torch.zeros(iml_target_size)
    iml = IML(iml_source_size)
    def iml_randomise():
        while(len(iml.pairs) < 32):
            source = torch.rand(iml_source_size)#/(ctrl.abs()/2+1)
            target = iml_target + torch.randn(iml_target_size)*2#/(z.abs()/2+1)
            print(source, target, iml_target)
            iml.add(source, target)
    # iml_randomise()
    # def iml_train():
    #     for i in range(32):
    #         source = torch.full((img.shape[0], img.shape[1]), i/32.0).flatten()
    #         target = torch.full((iml_target_size,), i/32.0)
    #         print(i, i/32.0)
    #         iml.add(source, target)
    w,h = 480,270#img.shape[0], img.shape[1]
    t_w, t_h = w//4, h//5
    lin_grad_x = np.linspace(1, 0, w, dtype=np.float32)
    lin_grad_y = np.linspace(1, 0, h, dtype=np.float32)[:,None]
    iml_sources = [
        np.full((h, w), 1.0),
        np.full((h, w), 0.0),
        np.tile(lin_grad_x, (h, 1)),
        np.tile(lin_grad_y, (1, w)),
        np.tile(lin_grad_x[::-1], (h, 1)),
        np.tile(lin_grad_y[::-1], (1, w)),
        np.dot(lin_grad_y, lin_grad_x[None,:]),
        np.dot(lin_grad_y, lin_grad_x[None,:])[:,::-1],
        np.dot(lin_grad_y[::-1], lin_grad_x[None,:]),
        np.dot(lin_grad_y[::-1], lin_grad_x[None,:])[:,::-1]]
    iml_targets = []
    def iml_train():
        for i, s in enumerate(iml_sources):
            source = torch.from_numpy(s.flatten())
            tiles = s.reshape(h//t_h, t_h, w//t_w, t_w)
            avg = np.mean(tiles, axis=(1,3))
            target = torch.from_numpy(avg.flatten())
            iml_targets.append(target)
            print(i, avg)
            iml.add(source, target)
    iml_train()
    
    iml_test_px = tol.Pixel.field(shape=(x, y))
    @ti.kernel
    def iml_test_kernel(source: ti.types.ndarray(dtype=ti.f32, ndim=2)):
        for i, j in ti.ndrange(x, y):
            px = source[j, i]
            iml_test_px.rgba[i, j] = ti.Vector([px,px,px,1.0])
    iml_source_count = fps*2
    iml_source_counter = 0
    iml_source_index = 0
    def iml_test():
        nonlocal iml_source_counter, iml_source_count, iml_sources, cv, factor, iml_target_norm, iml_source_index
        source = iml_sources[iml_source_index % len(iml_sources)]
        upscaled = cv.cv2_pyr_up(source.astype(np.float32), factor)
        iml_test_kernel(upscaled)

        iml_source = torch.from_numpy(source.flatten())
        iml_target[:] = torch.from_numpy(iml.map(iml_source, k=3))
        iml_target_norm_prev = iml_target_norm
        iml_target_norm = iml_norm(iml_target, -6, 6)
        iml_print(iml_target_norm)

        iml_source_counter += 1
        if iml_source_counter == iml_source_count:
            iml_source_counter = 0
            iml_source_index += 1

    def iml_print(tensor):
        print(" ".join([f"{i:.2f}" for i in tensor.tolist()]))

    def iml_norm(tensor, min=-1, max=1):
        return 2 * (tensor - min) / (max - min) - 1
    
    def iml_map():
        nonlocal img, iml_target_norm
        iml_source = torch.from_numpy(img.flatten())
        iml_target[:] = torch.from_numpy(iml.map(iml_source, k=2))#.float()
        iml_target_norm_prev = iml_target_norm
        iml_target_norm = iml_norm(iml_target, -6, 6)
        iml_print(iml_target_norm)
    iml_update = Updater(iml_map, 1)

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
    iml_target_counter = 0
    @osc_map.add(io=io, count=1, send_mode='broadcast')
    def ctrl() -> tuple[str, float]:
        nonlocal iml_target_norm, iml_target_size, iml_target_counter, iml_target_norm_prev
        key = "iml"+str(iml_target_counter)
        val = float(iml_target_norm[iml_target_counter])
        iml_target_counter += 1
        if iml_target_counter == iml_target_size:
            iml_target_counter = 0
        return [key, val]

    '''''''''''''''''''''''''''''''''''''''''''''
    Render
    '''''''''''''''''''''''''''''''''''''''''''''
    def render():
        nonlocal img, _img
        img, _img = process_cv(img, _img)
        osc_map()
        iml_update(img)
        pixels.clear()
        pixels.px.rgba = cv.px_rgba
    
    def render_test():
        iml_test()
        pixels.clear()
        pixels.px.rgba = iml_test_px.rgba
    
    def render_physarum():
        nonlocal img, _img
        img, _img = process_cv(img, _img)
        iml_update()
        physarum.deposit_px(cv.px_g, 0.1)
        physarum(particles)
        pixels.set(physarum.trail.px)
        pixels.decay()

    # tol.utils.render(render, pixels)
    tol.utils.render(render_test, pixels)
    # tol.utils.render(render_physarum, pixels)

if __name__=='__main__':
    run(main)
