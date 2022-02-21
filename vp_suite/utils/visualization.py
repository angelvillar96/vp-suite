import math
import os

import numpy as np
import torch
import matplotlib.pyplot as plt

COLORS = {
    "green": [0, 200, 0],
    "red": [150, 0, 0],
    "yellow": [100, 100, 0],
    "black": [0, 0, 0],
    "white": [255, 255, 255]
}  # r, g, b

def get_color_array(color):
    r_g_b = COLORS.get(color, COLORS["white"])
    return np.array(r_g_b, dtype=np.uint8)[np.newaxis, np.newaxis, np.newaxis, ...]

def add_border_around_vid(vid, c_and_l, b_width=10):

    _, h, w, _ = vid.shape
    color_bars_vertical = [np.tile(get_color_array(c), (l, h, b_width, 1)) for (c, l) in c_and_l]
    cbv = np.concatenate(color_bars_vertical, axis=0)

    color_bars_horizontal = [np.tile(get_color_array(c), (l, b_width, w + 2 * b_width, 1)) for (c, l) in c_and_l]
    cbh = np.concatenate(color_bars_horizontal, axis=0)

    vid = np.concatenate([cbv, vid, cbv], axis=-2)   # add bars in the width dim
    vid = np.concatenate([cbh, vid, cbh], axis=-3)   # add bars in the height dim
    return vid

def save_vid_vis(out_fp, context_frames, mode="gif", **trajs):

    trajs = {k: v for k, v in trajs.items() if v is not None}  # filter out 'None' trajs
    T, h, w, _ = list(trajs.values())[0].shape
    T_in, T_pred = context_frames, T-context_frames
    for key, traj in trajs.items():
        if "true_" in key.lower() or "gt_" in key.lower() or key.lower() == "gt":
            trajs[key] = add_border_around_vid(traj, [("green", T)], b_width=16)
        elif "seg" in key.lower():
            trajs[key] = add_border_around_vid(traj, [("yellow", T)], b_width=16)
        else:
            trajs[key] = add_border_around_vid(traj, [("green", T_in), ("red", T_pred)], b_width=16)

    if mode == "gif":  # gif visualizations with matplotlib  # TODO fix it
        try:
            from matplotlib import pyplot as PLT
            PLT.rcParams.update({'axes.titlesize': 'small'})
            from matplotlib.animation import FuncAnimation
        except ImportError:
            raise ImportError("importing from matplotlib failed "
                              "-> please install matplotlib or use the mp4-mode for visualization.")
        n_trajs = len(trajs)
        plt_scale = 0.01
        plt_cols = math.ceil(math.sqrt(n_trajs))
        plt_rows = math.ceil(n_trajs / plt_cols)
        plt_w = 1.2 * w * plt_scale * plt_cols
        plt_h = 1.4 * h * plt_scale * plt_rows
        fig = PLT.figure(figsize=(plt_w, plt_h), dpi=100)

        def update(t):
            for i, (name, traj) in enumerate(trajs.items()):
                PLT.subplot(plt_rows, plt_cols, i + 1)
                PLT.xticks([])
                PLT.yticks([])
                PLT.title(' '.join(name.split('_')).title())
                PLT.imshow(traj[t])

        anim = FuncAnimation(fig, update, frames=np.arange(T), interval=500)
        anim.save(out_fp, writer="imagemagick", dpi=200)
        PLT.close(fig)

    else:  # mp4 visualizations with opencv and moviepy
        try:
            from moviepy.editor import ImageSequenceClip
        except ImportError:
            raise ImportError("importing from moviepy failed"
                              " -> please install moviepy or use the gif-mode for visualization.")
        try:
            import cv2 as cv
        except ImportError:
            raise ImportError("importing cv2 failed"
                              " -> please install opencv-python (cv2) or use the gif-mode for visualization.")
        for name, traj in trajs.items():
            frames = list(traj)
            out_paths = []
            for t, frame in enumerate(frames):
                out_fn = f"{out_fp[:-4]}_{name}_t{t}.jpg"
                out_paths.append(out_fn)
                out_frame_BGR = frame[:, :, ::-1]
                cv.imwrite(out_fn, out_frame_BGR)
            clip = ImageSequenceClip(out_paths, fps=2)
            clip.write_videofile(f"{out_fp[:-4]}_{name}.mp4", fps=2)
            for out_fn in out_paths:
                os.remove(out_fn)


def visualize_vid(dataset, context_frames, pred_frames, pred_model, device,
                  out_path, vis_idx, mode="gif"):

    out_fn_template = "vis_{}." + mode
    data_unpack_config = {"device": device, "context_frames": context_frames, "pred_frames": pred_frames}

    if vis_idx is None or any([x >= len(dataset) for x in vis_idx]):
        raise ValueError(f"invalid vis_idx provided for visualization "
                         f"(dataset len = {len(dataset)}): {vis_idx}")

    pred_model.eval()
    for i, n in enumerate(vis_idx):

        # prepare input and ground truth sequence
        data = dataset[n]  # [T, c, h, w]
        if pred_model.NEEDS_COMPLETE_INPUT:
            input, _, actions = pred_model.unpack_data(data, data_unpack_config)
            input_vis = dataset.postprocess(input.clone().squeeze(dim=0))
        else:
            input, target, actions = pred_model.unpack_data(data, data_unpack_config)
            full = torch.cat([input.clone(), target.clone()], dim=1)
            input_vis = dataset.postprocess(full.squeeze(dim=0))

        # fwd
        if pred_model is None:
            raise ValueError("Need to provide a valid prediction model for visualization!")
        with torch.no_grad():
            pred, _ = pred_model(input, pred_frames, actions=actions)  # [1, T_pred, c, h, w]

        # assemble prediction
        if pred_model.NEEDS_COMPLETE_INPUT:  # replace original pred frames with actual prediction
            input_and_pred = input
            input_and_pred[:, -pred.shape[1]:] = pred
        else:  # concat context frames and prediction
            input_and_pred = torch.cat([input, pred], dim=1)  # [1, T, c, h, w]
        pred_vis = dataset.postprocess(input_and_pred.squeeze(dim=0))  # [T, h, w, c]

        # visualize
        out_filename = str(out_path / out_fn_template.format(str(i)))
        save_vid_vis(out_fp=out_filename, context_frames=context_frames, GT=input_vis, Pred=pred_vis, mode=mode)

    pred_model.train()

def save_diff_hist(diff, diff_id):
    avg_diff, min_diff, max_diff = np.average(diff), np.min(diff), np.max(diff)
    plt.hist(diff.flatten(), bins=1000, log=True)
    plt.suptitle(f"np.abs(their_pred - our_pred)\n"
                 f"min: {min_diff}, max: {max_diff}, avg: {avg_diff}")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.xscale("log")
    plt.yscale("log")
    plt.savefig(f"diff_{diff_id}.png")
