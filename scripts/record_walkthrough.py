"""
AtelierAI - Pure High-Quality Screen Recording Walkthrough (No TTS).

Captures a pristine, source-quality 1080p 30fps screen recording of the live AtelierAI application
using Playwright browser automation without any TTS synthetic audio.
Exported using NVIDIA NVENC high-profile visual encoding.
"""

import os
import sys
import time
import math
from pathlib import Path
import subprocess
from playwright.sync_api import sync_playwright

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEMO_DIR = PROJECT_ROOT / "demo"
RAW_DIR = DEMO_DIR / "raw"
FFMPEG_EXE = shutil.which("ffmpeg") or os.environ.get("FFMPEG_PATH") or r"C:\Users\Arnav\AppData\Local\CapCut\Apps\9.3.0.3970\ffmpeg.exe"

# Planned Scene Timings (Total: ~165s = 2m 45s, well within the 2:00-3:00 target window)
SCENE_PLAN = {
    "scene1": {"name": "Product Introduction", "duration": 20.0},
    "scene2": {"name": "Natural-Language Generation", "duration": 40.0},
    "scene3": {"name": "Multimodal CLIP Verification", "duration": 25.0},
    "scene4": {"name": "cGAN vs VAE Comparison", "duration": 35.0},
    "scene5": {"name": "Empirical Performance & Verification", "duration": 30.0},
    "scene6": {"name": "Architecture Insights & Outro", "duration": 18.0},
}


def capture_pure_browser_video() -> Path:
    """Launches Playwright and captures the live application session in pristine source quality."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for old_file in RAW_DIR.glob("*.webm"):
        try:
            old_file.unlink()
        except Exception:
            pass

    print("\n[1/3] Starting Playwright browser recording (1920x1080 @ 30fps, Source Quality)...")
    start_time = time.time()

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=[
                "--window-size=1920,1080",
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--force-device-scale-factor=1",
                "--font-render-hinting=medium",
                "--enable-font-antialiasing",
            ],
        )
        context = browser.new_context(
            viewport={"width": 1920, "height": 1080},
            record_video_dir=str(RAW_DIR),
            record_video_size={"width": 1920, "height": 1080},
        )
        page = context.new_page()

        def smooth_mouse_move(start_x, start_y, end_x, end_y, steps=25, delay=0.015):
            for i in range(steps + 1):
                t = i / steps
                t_smooth = (1 - math.cos(t * math.pi)) / 2
                cx = start_x + (end_x - start_x) * t_smooth
                cy = start_y + (end_y - start_y) * t_smooth
                page.mouse.move(cx, cy)
                time.sleep(delay)

        # -------------------------------------------------------------
        # SCENE 1: Product Introduction (0:00 - 0:20)
        # -------------------------------------------------------------
        print(f"  -> [Scene 1] {SCENE_PLAN['scene1']['name']} (20.0s)...")
        scene1_start = time.time()
        page.goto("http://localhost:5173")
        page.wait_for_selector("text=AtelierAI", timeout=10000)
        page.wait_for_timeout(1000)

        # Hover over Brand identity
        smooth_mouse_move(960, 540, 200, 32, steps=20)
        page.wait_for_timeout(3000)

        # Hover over GPU device health badge (RTX 3060 CUDA)
        smooth_mouse_move(200, 32, 1720, 32, steps=25)
        page.wait_for_timeout(3500)

        # Hover over Domain Scope banner
        smooth_mouse_move(1720, 32, 500, 100, steps=20)
        page.wait_for_timeout(4000)

        # Smooth mouse pass over navbar tabs
        smooth_mouse_move(500, 100, 750, 32, steps=20)
        page.wait_for_timeout(2000)
        smooth_mouse_move(750, 32, 900, 32, steps=15)
        page.wait_for_timeout(2000)
        smooth_mouse_move(900, 32, 1050, 32, steps=15)
        page.wait_for_timeout(2000)

        # Position cursor at prompt box
        smooth_mouse_move(1050, 32, 380, 260, steps=20)
        rem1 = SCENE_PLAN["scene1"]["duration"] - (time.time() - scene1_start)
        if rem1 > 0:
            time.sleep(rem1)

        # -------------------------------------------------------------
        # SCENE 2: Natural-Language Generation (0:20 - 1:00)
        # -------------------------------------------------------------
        print(f"  -> [Scene 2] {SCENE_PLAN['scene2']['name']} (40.0s)...")
        scene2_start = time.time()
        prompt_box = page.locator("#prompt-input")
        prompt_box.click()
        page.wait_for_timeout(600)
        prompt_box.fill("")
        page.wait_for_timeout(300)

        # Type text with natural human cadence
        target_prompt = "Generate four ankle boots."
        if hasattr(prompt_box, "press_sequentially"):
            prompt_box.press_sequentially(target_prompt, delay=60)
        else:
            prompt_box.type(target_prompt, delay=60)
        page.wait_for_timeout(1000)

        # Hover over cGAN toggle and strategy
        smooth_mouse_move(380, 260, 250, 360, steps=15)
        page.wait_for_timeout(1200)

        # Move to Synthesize Apparel button
        synth_btn = page.locator("button:has-text('Synthesize Apparel')")
        box = synth_btn.bounding_box()
        if box:
            smooth_mouse_move(250, 360, box["x"] + box["width"] / 2, box["y"] + box["height"] / 2, steps=20)
        page.wait_for_timeout(800)
        synth_btn.click()

        # Wait for live cGAN generation and oracle verification
        print("     Waiting for live cGAN synthesis on CUDA RTX 3060...")
        page.wait_for_selector("text=oracle match", timeout=15000)
        page.wait_for_timeout(2000)

        # Inspect generated ankle boot cards
        cards = page.locator(".group.relative")
        card_count = cards.count()
        print(f"     Generated {card_count} ankle boot cards.")
        for i in range(min(4, card_count)):
            c_box = cards.nth(i).bounding_box()
            if c_box:
                smooth_mouse_move(box["x"] if box else 500, box["y"] if box else 500, c_box["x"] + c_box["width"] / 2, c_box["y"] + c_box["height"] / 2, steps=18)
                page.wait_for_timeout(2000)

        # Hover over metadata summary badge (Class #9: Ankle boot, Qty: 4)
        smooth_mouse_move(960, 540, 1100, 240, steps=20)
        page.wait_for_timeout(3500)

        rem2 = SCENE_PLAN["scene2"]["duration"] - (time.time() - scene2_start)
        if rem2 > 0:
            time.sleep(rem2)

        # -------------------------------------------------------------
        # SCENE 3: Multimodal CLIP Verification (1:00 - 1:25)
        # -------------------------------------------------------------
        print(f"  -> [Scene 3] {SCENE_PLAN['scene3']['name']} (25.0s)...")
        scene3_start = time.time()
        # Scroll smoothly down to CLIP section
        page.evaluate("window.scrollTo({top: 450, behavior: 'smooth'})")
        page.wait_for_timeout(2500)

        # Hover over Raw Cosine Similarity metric
        smooth_mouse_move(1100, 240, 1150, 480, steps=20)
        page.wait_for_timeout(5000)

        # Hover over uncalibrated metric range label
        smooth_mouse_move(1150, 480, 1380, 480, steps=15)
        page.wait_for_timeout(5000)

        # Hover over the explicit domain shift limitation disclaimer
        smooth_mouse_move(1380, 480, 1100, 620, steps=20)
        page.wait_for_timeout(7000)

        rem3 = SCENE_PLAN["scene3"]["duration"] - (time.time() - scene3_start)
        if rem3 > 0:
            time.sleep(rem3)

        # -------------------------------------------------------------
        # SCENE 4: cGAN versus VAE Comparison (1:25 - 2:00)
        # -------------------------------------------------------------
        print(f"  -> [Scene 4] {SCENE_PLAN['scene4']['name']} (35.0s)...")
        scene4_start = time.time()
        # Scroll back to top
        page.evaluate("window.scrollTo({top: 0, behavior: 'smooth'})")
        page.wait_for_timeout(1000)

        # Click Model Comparison tab
        compare_tab = page.locator("button:has-text('Model Comparison')")
        c_box = compare_tab.bounding_box()
        if c_box:
            smooth_mouse_move(1100, 620, c_box["x"] + c_box["width"] / 2, c_box["y"] + c_box["height"] / 2, steps=25)
        page.wait_for_timeout(500)
        compare_tab.click()

        page.wait_for_selector("text=Conditional GAN vs Variational Autoencoder", timeout=10000)
        page.wait_for_timeout(1500)

        # Click Compare Synthesis button
        compare_btn = page.locator("button:has-text('Compare Synthesis')")
        cb_box = compare_btn.bounding_box()
        if cb_box:
            smooth_mouse_move(c_box["x"] if c_box else 700, c_box["y"] if c_box else 32, cb_box["x"] + cb_box["width"] / 2, cb_box["y"] + cb_box["height"] / 2, steps=20)
        page.wait_for_timeout(600)
        compare_btn.click()

        print("     Waiting for live side-by-side synthesis (cGAN vs VAE)...")
        page.wait_for_selector("text=Adversarial Generator", timeout=15000)
        page.wait_for_timeout(2000)

        # Hover over cGAN column (sharp edges)
        smooth_mouse_move(cb_box["x"] if cb_box else 900, cb_box["y"] if cb_box else 200, 600, 450, steps=25)
        page.wait_for_timeout(5000)

        # Hover over VAE column (smooth reconstructions)
        smooth_mouse_move(600, 450, 1300, 450, steps=25)
        page.wait_for_timeout(5000)

        # Scroll smoothly to bottom analysis cards
        page.evaluate("window.scrollTo({top: 500, behavior: 'smooth'})")
        page.wait_for_timeout(6000)

        rem4 = SCENE_PLAN["scene4"]["duration"] - (time.time() - scene4_start)
        if rem4 > 0:
            time.sleep(rem4)

        # -------------------------------------------------------------
        # SCENE 5: Empirical Performance & Verification (2:00 - 2:30)
        # -------------------------------------------------------------
        print(f"  -> [Scene 5] {SCENE_PLAN['scene5']['name']} (30.0s)...")
        scene5_start = time.time()
        page.evaluate("window.scrollTo({top: 0, behavior: 'smooth'})")
        page.wait_for_timeout(1000)

        # Click Performance & Benchmarks tab
        perf_tab = page.locator("button:has-text('Performance & Benchmarks')")
        p_box = perf_tab.bounding_box()
        if p_box:
            smooth_mouse_move(1300, 450, p_box["x"] + p_box["width"] / 2, p_box["y"] + p_box["height"] / 2, steps=25)
        page.wait_for_timeout(500)
        perf_tab.click()

        page.wait_for_selector("text=Empirical Benchmarks & Verification", timeout=10000)
        page.wait_for_timeout(2000)

        # Inspect headline metric cards
        # cGAN Match (97.2%)
        smooth_mouse_move(p_box["x"] if p_box else 800, p_box["y"] if p_box else 32, 450, 200, steps=20)
        page.wait_for_timeout(4000)

        # VAE Match (74.9%)
        smooth_mouse_move(450, 200, 750, 200, steps=20)
        page.wait_for_timeout(3500)

        # Qwen Standalone (90%) / Combined (100%)
        smooth_mouse_move(750, 200, 1050, 200, steps=20)
        page.wait_for_timeout(3500)

        # Scroll down to Per-Class Table & Weakest Classes
        page.evaluate("window.scrollTo({top: 480, behavior: 'smooth'})")
        page.wait_for_timeout(3000)

        # Hover over Weakest Classes card (Shirt 84%, Coat 96%, Pullover 97%)
        smooth_mouse_move(1050, 200, 1200, 520, steps=25)
        page.wait_for_timeout(6000)

        rem5 = SCENE_PLAN["scene5"]["duration"] - (time.time() - scene5_start)
        if rem5 > 0:
            time.sleep(rem5)

        # -------------------------------------------------------------
        # SCENE 6: Architecture Insights & Outro (2:30 - ~2:48)
        # -------------------------------------------------------------
        print(f"  -> [Scene 6] {SCENE_PLAN['scene6']['name']} (18.0s)...")
        scene6_start = time.time()
        page.evaluate("window.scrollTo({top: 0, behavior: 'smooth'})")
        page.wait_for_timeout(1000)

        # Click Model Insights tab
        insights_tab = page.locator("button:has-text('Model Insights')")
        i_box = insights_tab.bounding_box()
        if i_box:
            smooth_mouse_move(1200, 520, i_box["x"] + i_box["width"] / 2, i_box["y"] + i_box["height"] / 2, steps=25)
        page.wait_for_timeout(500)
        insights_tab.click()

        page.wait_for_selector("text=Model Insights & Theoretical Architecture", timeout=10000)
        page.wait_for_timeout(1500)

        # Click Layer III (Multi-Model Architecture: GPT, BERT, CLIP)
        layer3_btn = page.locator("button:has-text('GPT, BERT')")
        l3_box = layer3_btn.bounding_box()
        if l3_box:
            smooth_mouse_move(i_box["x"] if i_box else 900, i_box["y"] if i_box else 32, l3_box["x"] + l3_box["width"] / 2, l3_box["y"] + l3_box["height"] / 2, steps=20)
        page.wait_for_timeout(500)
        layer3_btn.click()
        page.wait_for_timeout(3500)

        # Click Layer I (Foundations: Generative vs Discriminative)
        layer1_btn = page.locator("button:has-text('Generative vs Discriminative')")
        l1_box = layer1_btn.bounding_box()
        if l1_box:
            smooth_mouse_move(l3_box["x"] if l3_box else 800, l3_box["y"] if l3_box else 140, l1_box["x"] + l1_box["width"] / 2, l1_box["y"] + l1_box["height"] / 2, steps=20)
        page.wait_for_timeout(500)
        layer1_btn.click()
        page.wait_for_timeout(3500)

        # Return to Studio tab for hero outro
        studio_tab = page.locator("button:has-text('Studio')")
        st_box = studio_tab.bounding_box()
        if st_box:
            smooth_mouse_move(l1_box["x"] if l1_box else 400, l1_box["y"] if l1_box else 140, st_box["x"] + st_box["width"] / 2, st_box["y"] + st_box["height"] / 2, steps=25)
        page.wait_for_timeout(500)
        studio_tab.click()
        page.wait_for_selector("text=AtelierAI", timeout=10000)
        smooth_mouse_move(st_box["x"] if st_box else 500, st_box["y"] if st_box else 32, 960, 540, steps=25)
        page.wait_for_timeout(4000)

        # Close video recording cleanly
        video = page.video
        context.close()
        browser.close()
        recorded_path = Path(video.path())
        total_rec_time = time.time() - start_time
        print(f"\n[Recorded WebM]: {recorded_path} ({recorded_path.stat().st_size / (1024*1024):.2f} MB, total time: {total_rec_time:.1f}s)")
        return recorded_path


def export_source_quality_mp4(webm_path: Path) -> Path:
    """
    Transcodes raw recording to high-profile, visually lossless H.264 MP4 with NO audio/TTS.
    Uses NVIDIA NVENC with slowest/highest-quality preset (p7) and low CQ for pure source preservation.
    """
    output_mp4 = DEMO_DIR / "AtelierAI_Product_Walkthrough.mp4"
    if output_mp4.exists():
        output_mp4.unlink()

    print("\n[2/3] Exporting source-quality pure MP4 (NVIDIA NVENC H.264, No Audio / No TTS)...")

    cmd = [
        FFMPEG_EXE,
        "-y",
        "-i", str(webm_path),
        "-an",  # Pure video, NO audio, NO TTS
        "-c:v", "h264_nvenc",
        "-preset", "p7",           # Slowest, highest encoding quality
        "-tune", "hq",             # High-quality tuning
        "-profile:v", "high",      # High profile
        "-rc", "vbr",              # Variable bitrate with target quality
        "-cq", "14",               # Visually lossless constant quality
        "-b:v", "25M",             # High bitrate for razor-sharp 1080p pixels
        "-maxrate", "35M",
        "-bufsize", "50M",
        "-pix_fmt", "yuv420p",
        "-r", "30",
        str(output_mp4),
    ]

    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        print("[WARN] High-bitrate NVENC failed, trying fallback encoder...")
        fallback_cmd = [
            FFMPEG_EXE,
            "-y",
            "-i", str(webm_path),
            "-an",
            "-c:v", "h264_mf",
            "-b:v", "20M",
            "-pix_fmt", "yuv420p",
            "-r", "30",
            str(output_mp4),
        ]
        res_fb = subprocess.run(fallback_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res_fb.returncode != 0:
            print("[ERROR] FFmpeg export failed:")
            print(res_fb.stderr)
            raise RuntimeError("FFmpeg export failed.")

    print(f"\n[3/3] Final Pure Video MP4 exported at: {output_mp4}")
    return output_mp4


def verify_pure_video(mp4_path: Path):
    """Verifies duration, resolution, stream specs, and pure video status."""
    print("\n=================================================================")
    print("             FINAL PURE DEMO VIDEO VERIFICATION                  ")
    print("=================================================================")
    if not mp4_path.exists():
        raise FileNotFoundError(f"Final video does not exist at {mp4_path}")

    file_size_mb = mp4_path.stat().st_size / (1024 * 1024)

    cmd = [FFMPEG_EXE, "-i", str(mp4_path)]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    output = res.stderr

    duration_str = "Unknown"
    resolution_str = "Unknown"
    video_codec = "Unknown"
    has_audio = False

    for line in output.splitlines():
        if "Duration:" in line:
            parts = line.split("Duration:")[1].split(",")[0].strip()
            duration_str = parts
        if "Stream #" in line and "Video:" in line:
            video_codec = line.split("Video:")[1].split()[0]
            if "1920x1080" in line:
                resolution_str = "1920x1080 (1080p Full HD, Source Quality)"
            else:
                resolution_str = line.split("Video:")[1]
        if "Audio:" in line:
            has_audio = True

    print(f"File Path:   {mp4_path}")
    print(f"File Size:   {file_size_mb:.2f} MB")
    print(f"Duration:    {duration_str}")
    print(f"Resolution:  {resolution_str}")
    print(f"Video Codec: {video_codec} (H.264 High Profile)")
    print(f"Audio Track: {'PRESENT (ERROR)' if has_audio else 'NONE (Pure Video Recording - No TTS)'}")
    print("=================================================================\n")


def main():
    webm_path = capture_pure_browser_video()
    mp4_path = export_source_quality_mp4(webm_path)
    verify_pure_video(mp4_path)


if __name__ == "__main__":
    main()
