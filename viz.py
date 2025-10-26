import numpy as np
from vispy import app, scene
from vispy.scene import Node
from vispy.scene.visuals import Sphere, Volume
from vispy.visuals.transforms import MatrixTransform, STTransform

from video_recorder import VideoRecorder

RECORD = False
ROTATION_SPEED = 2
GAMMA = 5
RAY_MARCHING_STEP_SIZE = 1
COLORMAP = "magma"
SHOW_ELECTRODES = True

voxels = np.load("voxels.npy").astype("float32")
ch_pos = np.load("ch_pos.npy").astype("float32")
voxels = (voxels - voxels.min()) / (voxels.max() - voxels.min())

canvas = scene.SceneCanvas(size=(800, 600), show=True, bgcolor="black")
view = canvas.central_widget.add_view()
holo = Volume(
    voxels[0],
    method="additive",
    clim=(0.0, 1.0),
    cmap=COLORMAP,
    relative_step_size=RAY_MARCHING_STEP_SIZE,
    gamma=GAMMA,
)
holo.transform = MatrixTransform()
center = np.array([holo.bounds(0), holo.bounds(1), holo.bounds(2)]).mean(axis=1)
holo.transform.translate(-center)
view.add(holo)

if RECORD:
    rec = VideoRecorder("vispy_record.mp4", size=canvas.size, fps=60)

if SHOW_ELECTRODES:
    elecs = Node()
    for ch in ch_pos:
        sphere = Sphere(radius=0.2, method="ico", subdivisions=2, color="yellow")
        sphere.mesh.set_gl_state("translucent", depth_test=False)
        sphere.transform = STTransform(translate=ch)
        sphere.parent = elecs
    elecs.transform = MatrixTransform()
    elecs.transform.translate(-center)
    view.add(elecs)

view.camera = "arcball"
view.camera.center = (0, 0, 0)


def update(ev):
    i = ev.count % len(voxels)
    holo.set_data(voxels[i])
    holo.transform.rotate(ev.dt * ROTATION_SPEED, (0, 0, 1))
    holo.update()

    if SHOW_ELECTRODES:
        elecs.transform.rotate(ev.dt * ROTATION_SPEED, (0, 0, 1))
        elecs.update()

    if RECORD:
        rec.write(canvas.render(alpha=False))


timer = app.Timer(1 / 60, connect=update, start=True)


def on_close(ev):
    if RECORD:
        print("Finalizing recording...")
        rec.close()


canvas.events.close.connect(on_close)
app.run()
