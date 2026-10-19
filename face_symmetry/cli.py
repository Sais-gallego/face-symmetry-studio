"""Command Line Interface for Face Symmetry Studio."""

import os
import sys
import argparse
import cv2
import numpy as np
from .generator import SymmetryStudio

def create_synthetic_portrait(width: int = 500, height: int = 600) -> np.ndarray:
    """Generate an artistic synthetic face for demonstrations and testing."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:] = (235, 220, 205)  # Light neutral background
    
    # Head contour (slightly asymmetric ellipse)
    center = (width // 2, height // 2)
    cv2.ellipse(img, center, (140, 190), 0, 0, 360, (190, 170, 150), -1)
    
    # Left eye (y=260, x=190)
    cv2.circle(img, (190, 260), 16, (60, 50, 40), -1)
    cv2.circle(img, (190, 260), 6, (20, 15, 10), -1)
    
    # Right eye (y=258, x=310 - slightly asymmetric by 2px)
    cv2.circle(img, (310, 258), 16, (60, 50, 40), -1)
    cv2.circle(img, (310, 258), 6, (20, 15, 10), -1)
    
    # Eyebrows
    cv2.line(img, (170, 235), (215, 230), (50, 40, 30), 4)
    cv2.line(img, (285, 232), (330, 236), (50, 40, 30), 4)
    
    # Nose
    cv2.line(img, (250, 260), (252, 330), (160, 140, 120), 3)
    cv2.line(img, (252, 330), (265, 340), (160, 140, 120), 3)
    
    # Mouth
    cv2.ellipse(img, (250, 395), (45, 15), 0, 0, 180, (140, 70, 70), -1)
    
    # Add subtle skin texture noise
    noise = np.random.normal(0, 5, img.shape).astype(np.int16)
    noisy_img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return noisy_img

def main():
    parser = argparse.ArgumentParser(
        description="Face Symmetry Studio: Biometric Facial Symmetry Analysis & Mirror Cloning Engine"
    )
    parser.add_argument("--image", "-i", type=str, help="Path to input portrait image (JPG, PNG)")
    parser.add_argument("--output-dir", "-o", type=str, default="results", help="Directory to save output files")
    parser.add_argument("--demo", action="store_true", help="Run demonstration using built-in synthetic portrait")
    parser.add_argument("--web", action="store_true", help="Launch interactive Flask Web Studio")
    parser.add_argument("--port", type=int, default=5050, help="Web server port (default 5050)")
    
    args = parser.parse_args()
    
    if args.web:
        from .app import run_app
        run_app(port=args.port)
        return

    if args.demo or not args.image:
        print("[*] Running in DEMO mode with synthetic portrait...")
        input_image = create_synthetic_portrait()
        base_name = "demo_portrait"
    else:
        if not os.path.exists(args.image):
            print(f"[!] Error: File not found: {args.image}", file=sys.stderr)
            sys.exit(1)
        input_image = cv2.imread(args.image)
        if input_image is None:
            print(f"[!] Error: Failed to decode image: {args.image}", file=sys.stderr)
            sys.exit(1)
        base_name = os.path.splitext(os.path.basename(args.image))[0]

    os.makedirs(args.output_dir, exist_ok=True)
    studio = SymmetryStudio()
    
    print("[*] Analyzing facial geometry, midline sagittal axis, and bilateral balance...")
    results = studio.process_portrait(input_image)
    metrics = results["metrics"]
    
    print("\n" + "=" * 50)
    print("      BIOMETRIC SYMMETRY ANALYSIS REPORT")
    print("=" * 50)
    print(f"• Overall Symmetry Index:     {metrics.overall_symmetry_score}%")
    print(f"• Pixel Similarity Score:    {metrics.pixel_similarity_score}%")
    print(f"• Ocular Balance:             {metrics.ocular_symmetry_score}%")
    print(f"• Golden Ratio Score:         {metrics.golden_ratio_score}%")
    print("=" * 50)
    
    collage_path = os.path.join(args.output_dir, f"{base_name}_symmetry_analysis.png")
    left_path = os.path.join(args.output_dir, f"{base_name}_left_symmetric.png")
    right_path = os.path.join(args.output_dir, f"{base_name}_right_symmetric.png")
    heat_path = os.path.join(args.output_dir, f"{base_name}_asymmetry_heatmap.png")
    
    cv2.imwrite(collage_path, results["collage"])
    cv2.imwrite(left_path, results["left_symmetric"])
    cv2.imwrite(right_path, results["right_symmetric"])
    cv2.imwrite(heat_path, results["heatmap"])
    
    print(f"[+] 4-Panel Diagnostic Collage saved to: {collage_path}")
    print(f"[+] Left-Symmetric Clone saved to:      {left_path}")
    print(f"[+] Right-Symmetric Clone saved to:     {right_path}")
    print(f"[+] Asymmetry Heatmap saved to:          {heat_path}")

if __name__ == "__main__":
    main()
