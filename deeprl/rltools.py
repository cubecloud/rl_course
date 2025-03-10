import os
import imageio
import PIL
import shutil as sh
from glob import glob
from base64 import b64encode

__version__ = 0.007


def saveanimation(frames, _path_filename="./movie.gif"):
    """
    This method ,given the frames of images make the gif and save it in the folder

    params:
        frames:method takes in the array or np.array of images
        address:(optional)given the address/location saves the gif on that location
                otherwise save it to default address './movie.gif'

    return :
        none
    """
    imageio.mimsave(_path_filename, frames)


def save_mp4(eps_frames, path_filename, fps=25):
    eps_frame_dir = 'episode_frames'
    os.mkdir(eps_frame_dir)

    for i, frame in enumerate(eps_frames):
        PIL.Image.fromarray(frame).save(os.path.join(eps_frame_dir, f'frame-{i + 1}.png'))

    os.system(f'ffmpeg -v 0 -r {fps} -i {eps_frame_dir}/frame-%1d.png -vcodec libx264 -b 10M -y "{path_filename}"');
    sh.rmtree(eps_frame_dir)


def show_records(records_path):
    record_paths = glob(os.path.join(records_path, "*.mp4"))
    html_str = ''
    for i, record_path in enumerate(record_paths):
        mp4 = open(record_path, 'rb').read()
        data = f"data:video/mp4;base64,{b64encode(mp4).decode()}"
        html_str += f'EPISODE # {i + 1}<br><video width=500 controls><source src="{data}" type="video/mp4"></video><br><br>'
    return html_str


def show_mp4(path_filename):
    mp4 = open(path_filename, 'rb').read()
    data = f"data:video/mp4;base64,{b64encode(mp4).decode()}"
    html_str = f'<br><video width=500 controls><source src="{data}" type="video/mp4"></video><br><br>'
    return html_str
