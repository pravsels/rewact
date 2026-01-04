#!/usr/bin/env python

"""
Visualize SAM 3 'focus' on a LeRobot dataset episode.
This script demonstrates how SAM 3 segments relevant objects based on a task description.

Example:
    python scripts/visualise_sam3_focus.py \
        --dataset-repo-id "danaaubakirova/so100_task_2" \
        --episode-id 0 \
        --sam3-weights /path/to/sam3.pt \
        --task-description "stack the cubes" \
        --output outputs/sam3_focus_ep0.mp4
"""

import argparse
import os
import sys
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from tqdm import tqdm
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# Add sam3 to path - assumed to be a sibling of the current repo
SAM3_PATH = Path(__file__).resolve().parent.parent.parent / "sam3"
if str(SAM3_PATH) not in sys.path:
    sys.path.append(str(SAM3_PATH))

try:
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.data_misc import (
        BatchedDatapoint, 
        FindStage, 
        BatchedFindTarget, 
        BatchedInferenceMetadata
    )
    from sam3.model.geometry_encoders import Prompt
except ImportError as e:
    print(f"Error: Failed to import SAM 3 components: {e}")
    print(f"SAM3_PATH: {SAM3_PATH}")
    print("Please ensure the sam3 repository is installed or in the correct path.")
    import traceback
    traceback.print_exc()
    sys.exit(1)

def preprocess_image(img_tensor, target_res=1008):
    """
    SAM3 expects 1008x1008 square inputs normalized to [-1, 1].
    img_tensor: (C, H, W) in [0, 1]
    """
    c, h, w = img_tensor.shape
    scale = float(target_res) / float(max(h, w))
    new_h, new_w = int(round(h * scale)), int(round(w * scale))
    
    # Resize
    img = F.interpolate(img_tensor.unsqueeze(0), size=(new_h, new_w), mode="bilinear", align_corners=False)
    
    # Pad to square (center pad)
    pad_h, pad_w = target_res - new_h, target_res - new_w
    img = F.pad(img, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=0.5)
    
    # Normalize to [-1, 1]
    return (img - 0.5) / 0.5

def overlay_mask(image, mask, color=(0, 255, 0), alpha=0.4):
    """Overlays a semi-transparent color mask on a BGR image."""
    mask_visual = np.zeros_like(image)
    mask_visual[mask > 0] = color
    return cv2.addWeighted(image, 1.0, mask_visual, alpha, 0)

def main():
    parser = argparse.ArgumentParser(description="Visualize SAM 3 focus on LeRobot dataset")
    parser.add_argument("--dataset-repo-id", type=str, required=True, help="LeRobot dataset repo ID")
    parser.add_argument("--episode-id", type=int, required=True, help="Episode index to visualize")
    parser.add_argument("--sam3-weights", type=str, required=True, help="Path to SAM 3 checkpoint (.pt)")
    parser.add_argument("--task-description", type=str, default=None, help="Manual task description (optional)")
    parser.add_argument("--output", type=str, default="outputs/sam3_focus.mp4", help="Output video path")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--fps", type=int, default=None, help="Override FPS for output video")
    args = parser.parse_args()

    # 1. Load SAM 3 Model
    print(f"Loading SAM 3 model from {args.sam3_weights}...")
    model = build_sam3_image_model(
        checkpoint_path=args.sam3_weights,
        device=args.device,
        enable_segmentation=True,
        load_from_HF=False
    )
    model.eval()

    # 2. Load Dataset
    print(f"Loading dataset: {args.dataset_repo_id}")
    dataset = LeRobotDataset(args.dataset_repo_id)
    
    # Get task description if not provided
    task = args.task_description
    if task is None:
        # Try to find task in dataset meta or use a default
        task = "the objects involved in the task"
        print(f"No task description provided, using default: '{task}'")

    # 3. Process Episode
    episode_indices = dataset.hf_dataset.filter(lambda x: x["episode_index"] == args.episode_id)["index"]
    if len(episode_indices) == 0:
        print(f"Error: Episode {args.episode_id} not found in dataset.")
        sys.exit(1)
        
    print(f"Processing episode {args.episode_id} ({len(episode_indices)} frames)")

    frames_out = []
    
    for idx in tqdm(episode_indices):
        frame_data = dataset[idx.item()]
        
        # Get the first camera image (usually 'observation.image' or similar)
        img_key = [k for k in frame_data.keys() if "image" in k and "past" not in k][0]
        img_tensor = frame_data[img_key] # (C, H, W) in [0, 1]
        
        # Original image for visualization (BGR for OpenCV)
        orig_img = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        orig_img = cv2.cvtColor(orig_img, cv2.COLOR_RGB2BGR)
        h, w = orig_img.shape[:2]

        # Prepare for SAM 3
        # Use exact preprocessing SAM 3 expects
        input_img = preprocess_image(img_tensor).to(args.device)
        
        # Create minimal grounding structures
        stage = FindStage(
            img_ids=torch.tensor([0], device=args.device),
            text_ids=torch.tensor([0], device=args.device),
            input_boxes=torch.zeros((1, 0, 4), device=args.device),
            input_boxes_mask=torch.ones((1, 0), dtype=torch.bool, device=args.device),
            input_boxes_label=torch.zeros((1, 0), dtype=torch.long, device=args.device),
            input_points=torch.zeros((1, 0, 2), device=args.device),
            input_points_mask=torch.ones((1, 0), dtype=torch.bool, device=args.device),
        )

        batch = BatchedDatapoint(
            img_batch=input_img,
            find_text_batch=[task],
            find_inputs=[stage],
            find_targets=[], 
            find_metadatas=[] 
        )

        with torch.inference_mode():
            # forward() returns a list of stages, we take the last stage's output
            # SAM3 output is a list of results per stage, we take the last step result
            outputs_list = model(batch)
            outputs = outputs_list[-1][-1]
            
            # Extract masks and scores
            pred_logits = outputs["pred_logits"] # (num_prompts, num_queries, 1)
            pred_masks = outputs["pred_masks"]   # (num_prompts, num_queries, H_m, W_m)
            
            # For each prompt (we only have one), find the best scoring query
            scores = pred_logits[0, :, 0].sigmoid()
            best_query_idx = scores.argmax()
            
            # Visualize the best mask
            best_mask = (pred_masks[0, best_query_idx] > 0).cpu().numpy().astype(np.uint8)
            
            # Resize mask back to original resolution if it differs
            if best_mask.shape != (h, w):
                best_mask = cv2.resize(best_mask, (w, h), interpolation=cv2.INTER_NEAREST)
            
            vis_img = overlay_mask(orig_img, best_mask)
            
            # Add UI overlays
            cv2.putText(vis_img, f"Task: {task}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(vis_img, f"Confidence: {scores[best_query_idx]:.2f}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            frames_out.append(vis_img)

    # 4. Save Video
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fps = args.fps if args.fps is not None else dataset.fps
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(args.output, fourcc, fps, (w, h))
    for f in frames_out:
        out_video.write(f)
    out_video.release()
    print(f"Visualization saved to {args.output}")

if __name__ == "__main__":
    main()

