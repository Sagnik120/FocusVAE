"""
Gradio demo: upload an image, type a query, see:
  - the relevance heatmap FocusVAE computed for that query
  - FocusVAE's reconstruction vs the baseline VanillaVAE's reconstruction
  - estimated bits-per-pixel for each

Usage:
    python demo/app.py --vanilla_ckpt checkpoints/vanilla/best.pt \
                        --focus_ckpt checkpoints/focus/best.pt
"""
import argparse
import sys
from pathlib import Path

import gradio as gr
import matplotlib.cm as cm
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from evaluate import build_from_ckpt  # noqa: E402
from losses import estimate_bpp  # noqa: E402
from utils import get_device  # noqa: E402

DEVICE = get_device()
VANILLA, FOCUS, IMAGE_SIZE = None, None, 64


def load_models(vanilla_ckpt, focus_ckpt):
    global VANILLA, FOCUS, IMAGE_SIZE
    VANILLA, vcfg = build_from_ckpt(vanilla_ckpt, DEVICE)
    FOCUS, fcfg = build_from_ckpt(focus_ckpt, DEVICE)
    IMAGE_SIZE = vcfg["image_size"]


def to_pil(tensor):
    arr = (tensor.clamp(0, 1).cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    return Image.fromarray(arr)


def heatmap_overlay(image_pil, relevance_2d):
    """relevance_2d: small grid (e.g. 8x8) in [0,1] -> upsampled colored overlay."""
    rel = torch.tensor(relevance_2d).unsqueeze(0).unsqueeze(0)
    rel_up = torch.nn.functional.interpolate(rel, size=image_pil.size[::-1],
                                              mode="bilinear", align_corners=False)
    rel_up = rel_up.squeeze().numpy()
    colored = (cm.inferno(rel_up)[:, :, :3] * 255).astype(np.uint8)
    base = np.array(image_pil).astype(np.float32)
    blended = (0.5 * base + 0.5 * colored).astype(np.uint8)
    return Image.fromarray(blended)


@torch.no_grad()
def run(image, query):
    if image is None or not query:
        return None, None, None, "Upload an image and type a query."

    tf = transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor()])
    x = tf(image.convert("RGB")).unsqueeze(0).to(DEVICE)

    v_out = VANILLA(x)
    f_out = FOCUS(x, queries=[query])

    v_bpp = estimate_bpp(v_out["mu"], v_out["logvar"], x.shape[-2:]).item()
    f_bpp = estimate_bpp(f_out["mu"], f_out["logvar"], x.shape[-2:]).item()

    resized_input = to_pil(x[0])
    v_recon = to_pil(v_out["recon"][0])
    f_recon = to_pil(f_out["recon"][0])
    relevance = f_out["relevance"][0].cpu().numpy()
    overlay = heatmap_overlay(resized_input, relevance)

    caption = (f"VanillaVAE bpp: {v_bpp:.4f}   |   FocusVAE bpp: {f_bpp:.4f}\n"
               f"(FocusVAE should reconstruct the query-relevant region better "
               f"at a similar bit budget.)")
    return overlay, v_recon, f_recon, caption


def build_ui():
    with gr.Blocks(title="FocusVAE Demo") as demo:
        gr.Markdown("# FocusVAE — Query-Adaptive Semantic Compression\n"
                     "Upload a face image and ask a question about it "
                     "(e.g. *'is the person smiling?'*, *'is the person wearing glasses?'*). "
                     "FocusVAE spends more of its latent bits on the region your query cares about.")
        with gr.Row():
            image_in = gr.Image(type="pil", label="Input image")
            query_in = gr.Textbox(label="Query", placeholder="is the person smiling?")
        run_btn = gr.Button("Run")
        with gr.Row():
            heatmap_out = gr.Image(label="Relevance heatmap (FocusVAE)")
            vanilla_out = gr.Image(label="VanillaVAE reconstruction")
            focus_out = gr.Image(label="FocusVAE reconstruction")
        caption_out = gr.Textbox(label="bpp comparison", interactive=False)

        run_btn.click(run, inputs=[image_in, query_in],
                       outputs=[heatmap_out, vanilla_out, focus_out, caption_out])
    return demo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vanilla_ckpt", type=str, default="checkpoints/vanilla/best.pt")
    ap.add_argument("--focus_ckpt", type=str, default="checkpoints/focus/best.pt")
    ap.add_argument("--share", action="store_true")
    args = ap.parse_args()

    load_models(args.vanilla_ckpt, args.focus_ckpt)
    demo = build_ui()
    demo.launch(share=args.share)


if __name__ == "__main__":
    main()
