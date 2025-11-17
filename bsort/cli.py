"""
bsort CLI - Bottle Cap Sorting System
Command-line interface for training and inference
"""

import logging
import sys
import time
from pathlib import Path

import cv2
import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from bsort import __version__
from bsort.config import load_config

# Setup rich console
console = Console()
app = typer.Typer(
    name="bsort",
    help="Real-time bottle cap detection and sorting system for Raspberry Pi 5",
    add_completion=False,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True, console=console)],
)
logger = logging.getLogger(__name__)


def version_callback(value: bool):
    """Show version and exit."""
    if value:
        console.print(f"[bold cyan]bsort[/bold cyan] version {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        None,
        "--version",
        "-v",
        callback=version_callback,
        is_eager=True,
        help="Show version and exit",
    ),
):
    """
    bsort - Real-time bottle cap detection optimized for Raspberry Pi 5.

    Target latency: 5-10ms inference using YOLO11n + INT8 quantization.
    """
    pass


@app.command()
def train(
    data_dir: str = typer.Option(
        "data/processed",
        "--data-dir",
        "-d",
        help="Directory containing train/val/test splits",
    ),
    config_path: str = typer.Option(
        "settings.yaml",
        "--config",
        "-c",
        help="Path to configuration file",
    ),
    output_dir: str = typer.Option(
        "runs/train",
        "--output",
        "-o",
        help="Output directory for training artifacts",
    ),
    export_onnx: bool = typer.Option(
        True,
        "--export-onnx/--no-export-onnx",
        help="Export model to ONNX after training",
    ),
    quantize: bool = typer.Option(
        True,
        "--quantize/--no-quantize",
        help="Apply INT8 quantization (critical for RPi5 performance)",
    ),
    resume: bool = typer.Option(
        False,
        "--resume",
        "-r",
        help="Resume from last checkpoint",
    ),
):
    """
    Train YOLO11n bottle cap detector with W&B tracking.

    Example:
        bsort train --data-dir data/processed --export-onnx --quantize
    """
    console.print("[bold cyan]Starting YOLO11n Training Pipeline[/bold cyan]")
    console.print("=" * 60)

    try:
        from bsort.train import train_model

        # Verify data directory exists
        if not Path(data_dir).exists():
            console.print(f"[bold red]Error:[/bold red] Data directory not found: {data_dir}")
            console.print("\nPlease run auto-labeling first:")
            console.print(
                "  python scripts/prepare_data.py --image-dir <path> --label-dir <path> --output-dir <path>"
            )
            raise typer.Exit(1)

        # Train
        results = train_model(
            config_path=config_path,
            data_dir=data_dir,
            output_dir=output_dir,
            export_onnx=export_onnx,
            quantize=quantize,
        )

        # Display results
        console.print("\n[bold green]✓ Training Complete![/bold green]")

        table = Table(title="Model Artifacts")
        table.add_column("Type", style="cyan")
        table.add_column("Path", style="green")

        for model_type, path in results.items():
            table.add_row(model_type, path)

        console.print(table)

        if quantize:
            console.print(
                "\n[bold yellow]Note:[/bold yellow] INT8 quantized model provides 3-4x speedup on RPi5!"
            )

    except Exception as e:
        console.print(f"[bold red]Training failed:[/bold red] {e}")
        logger.exception("Training error")
        raise typer.Exit(1)


@app.command()
def infer(
    model_path: str = typer.Option(
        "data/models/best_int8.onnx",
        "--model",
        "-m",
        help="Path to ONNX model (preferably INT8 quantized)",
    ),
    config_path: str = typer.Option(
        "settings.yaml",
        "--config",
        "-c",
        help="Path to configuration file",
    ),
    display: bool = typer.Option(
        True,
        "--display/--no-display",
        help="Display inference results",
    ),
    save_video: str = typer.Option(
        None,
        "--save-video",
        help="Save output video to file",
    ),
    monitor_temp: bool = typer.Option(
        True,
        "--monitor-temp/--no-monitor-temp",
        help="Monitor CPU temperature (RPi5)",
    ),
):
    """
    Run real-time inference using multiprocessing pipeline.

    This command uses a producer-consumer architecture to bypass Python's GIL:
    - Camera process: Captures frames -> Shared Memory
    - Inference process: ONNX Runtime inference
    - Display process: Visualization

    Example:
        bsort infer --model data/models/best_int8.onnx --display
    """
    console.print("[bold cyan]Real-time Inference Pipeline[/bold cyan]")
    console.print("=" * 60)

    # Verify model exists
    if not Path(model_path).exists():
        console.print(f"[bold red]Error:[/bold red] Model not found: {model_path}")
        raise typer.Exit(1)

    try:
        from bsort.infer import RealtimeInferencePipeline

        # Load config
        config = load_config(config_path)

        # Initialize pipeline
        pipeline = RealtimeInferencePipeline(config=config, model_path=model_path)

        console.print(f"[green]Model:[/green] {model_path}")
        console.print(
            f"[green]Input size:[/green] {config.get('inference.input_size', 640)}x{config.get('inference.input_size', 640)}"
        )
        console.print("[green]Starting pipeline...[/green]\n")

        # Start pipeline
        pipeline.start()

        # Video writer
        video_writer = None
        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(
                save_video,
                fourcc,
                30.0,
                tuple(config.get("inference.camera.resolution", [1280, 720])),
            )

        # FPS calculation
        fps_history = []
        frame_count = 0

        # Temperature monitoring
        def get_temperature():
            """Read CPU temperature on Raspberry Pi."""
            try:
                with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
                    temp = float(f.read()) / 1000.0
                return temp
            except:
                return None

        console.print("[yellow]Press 'q' to quit[/yellow]\n")

        try:
            while True:
                # Get results
                result = pipeline.get_results(timeout=1.0)

                if result is None:
                    continue

                detections = result["detections"]
                inference_time = result["inference_time"]

                # Calculate FPS
                fps = 1000.0 / inference_time if inference_time > 0 else 0
                fps_history.append(fps)
                if len(fps_history) > 30:
                    fps_history.pop(0)
                avg_fps = sum(fps_history) / len(fps_history)

                frame_count += 1

                # Display stats every 30 frames
                if frame_count % 30 == 0:
                    stats_table = Table(show_header=False, box=None)
                    stats_table.add_row("Inference Time", f"{inference_time:.2f} ms")
                    stats_table.add_row("FPS", f"{avg_fps:.1f}")
                    stats_table.add_row("Detections", str(len(detections)))

                    if monitor_temp:
                        temp = get_temperature()
                        if temp:
                            temp_color = "red" if temp > 75 else "yellow" if temp > 65 else "green"
                            stats_table.add_row(
                                "CPU Temp", f"[{temp_color}]{temp:.1f}°C[/{temp_color}]"
                            )

                    console.print(stats_table)

                    # Class distribution
                    if detections:
                        class_counts = {}
                        for det in detections:
                            class_counts[det.class_name] = class_counts.get(det.class_name, 0) + 1

                        console.print("\nDetected classes:")
                        for class_name, count in class_counts.items():
                            console.print(f"  {class_name}: {count}")
                    console.print("")

                # Check if we're hitting target latency
                if frame_count == 100:
                    if avg_fps < 100:  # 10ms target
                        console.print(
                            f"[yellow]Warning:[/yellow] Average FPS ({avg_fps:.1f}) below target (100 FPS / 10ms)"
                        )
                        console.print(
                            "Consider: reducing input size, ensuring INT8 model, active cooling"
                        )

        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted by user[/yellow]")

        finally:
            pipeline.stop()
            if video_writer:
                video_writer.release()

    except Exception as e:
        console.print(f"[bold red]Inference failed:[/bold red] {e}")
        logger.exception("Inference error")
        raise typer.Exit(1)


@app.command()
def benchmark(
    model_path: str = typer.Option(
        ...,
        "--model",
        "-m",
        help="Path to ONNX model",
    ),
    config_path: str = typer.Option(
        "settings.yaml",
        "--config",
        "-c",
        help="Path to configuration file",
    ),
    num_iterations: int = typer.Option(
        100,
        "--iterations",
        "-n",
        help="Number of inference iterations",
    ),
):
    """
    Benchmark model inference performance.

    Useful for validating that INT8 quantization achieves 5-10ms target.

    Example:
        bsort benchmark --model data/models/best_int8.onnx --iterations 100
    """
    console.print("[bold cyan]Model Benchmark[/bold cyan]")
    console.print("=" * 60)

    if not Path(model_path).exists():
        console.print(f"[bold red]Error:[/bold red] Model not found: {model_path}")
        raise typer.Exit(1)

    try:
        import numpy as np

        from bsort.infer import ONNXInferenceEngine

        config = load_config(config_path)

        # Initialize engine
        engine = ONNXInferenceEngine(
            model_path=model_path,
            input_size=config.get("inference.input_size", 640),
        )

        # Create dummy input
        input_size = config.get("inference.input_size", 640)
        dummy_image = np.random.randint(0, 255, (input_size, input_size, 3), dtype=np.uint8)

        console.print(f"Model: {model_path}")
        console.print(f"Input size: {input_size}x{input_size}")
        console.print(f"Iterations: {num_iterations}\n")

        # Warmup
        console.print("Warming up...")
        for _ in range(10):
            engine.infer(dummy_image)

        # Benchmark
        console.print("Benchmarking...")
        times = []
        for i in range(num_iterations):
            _, inference_time = engine.infer(dummy_image)
            times.append(inference_time)

            if (i + 1) % 20 == 0:
                console.print(f"Progress: {i+1}/{num_iterations}")

        # Statistics
        times = np.array(times)
        mean_time = np.mean(times)
        std_time = np.std(times)
        min_time = np.min(times)
        max_time = np.max(times)
        p50 = np.percentile(times, 50)
        p95 = np.percentile(times, 95)
        p99 = np.percentile(times, 99)

        # Results table
        results_table = Table(title="Benchmark Results")
        results_table.add_column("Metric", style="cyan")
        results_table.add_column("Value", style="green")

        results_table.add_row("Mean", f"{mean_time:.2f} ms")
        results_table.add_row("Std Dev", f"{std_time:.2f} ms")
        results_table.add_row("Min", f"{min_time:.2f} ms")
        results_table.add_row("Max", f"{max_time:.2f} ms")
        results_table.add_row("P50 (Median)", f"{p50:.2f} ms")
        results_table.add_row("P95", f"{p95:.2f} ms")
        results_table.add_row("P99", f"{p99:.2f} ms")
        results_table.add_row("FPS (from mean)", f"{1000/mean_time:.1f}")

        console.print("\n")
        console.print(results_table)

        # Target analysis
        console.print("\n[bold]Target Analysis:[/bold]")
        if mean_time <= 10:
            console.print("[bold green]✓ EXCELLENT[/bold green] - Meets 5-10ms target!")
        elif mean_time <= 15:
            console.print(
                "[bold yellow]⚠ GOOD[/bold yellow] - Close to target, consider optimizations"
            )
        else:
            console.print("[bold red]✗ NEEDS IMPROVEMENT[/bold red]")
            console.print("Recommendations:")
            console.print("  1. Ensure INT8 quantization is applied")
            console.print("  2. Reduce input size (640 -> 416)")
            console.print("  3. Use active cooling on RPi5")
            console.print("  4. Verify XNNPACK is enabled in ONNX Runtime")

    except Exception as e:
        console.print(f"[bold red]Benchmark failed:[/bold red] {e}")
        logger.exception("Benchmark error")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
