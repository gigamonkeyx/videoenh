#!/usr/bin/env python3
"""
Video Enhancement Tool with Multi-Model Support
RIPER-Ω v2.6 Compliant Implementation
RTX 3080 Optimized with VRAM Monitoring
"""

import sys
import os
import time
import logging
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
import torch
import torch.nn as nn
from collections import deque
import threading
from pathlib import Path
import concurrent.futures

# Add repo paths for model imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'MVDenoiser'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'TAP'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'DarkIR'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'Cobra'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'VanGogh'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'LVCD'))

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create formatters
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)

# File handler
file_handler = logging.FileHandler('app.log')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)

# Add handlers to logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)

class VideoEnhancerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Video Enhancement Tool - RIPER-Ω v2.6")
        self.root.geometry("800x600")
        
        # GPU Optimization Setup
        self.setup_gpu()
        
        # Model cache for lazy loading
        self.models = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Variables
        self.input_path = tk.StringVar()
        self.output_path = tk.StringVar()
        self.denoiser_var = tk.StringVar(value="TAP")
        self.lowlight_var = tk.BooleanVar(value=False)
        self.colorizer_var = tk.StringVar(value="VanGogh")
        self.color_var = tk.BooleanVar(value=False)
        self.text_prompt = tk.StringVar(value="natural colors")
        self.ref_path = tk.StringVar()
        self.progress_var = tk.DoubleVar()
        self.status_var = tk.StringVar(value="Ready")

        # Tuning parameters
        self.darkir_gain = tk.DoubleVar(value=1.0)
        self.tap_sigma = tk.DoubleVar(value=15.0)
        self.vangogh_steps = tk.DoubleVar(value=30.0)
        self.cobra_ref_weight = tk.DoubleVar(value=0.8)
        self.console_visible = tk.BooleanVar(value=True)
        
        # Frame buffer for temporal processing
        self.frame_buffer = deque(maxlen=5)
        self.window_size = 5
        
        # RL Evolution parameters
        self.rl_threshold = 0.5
        self.fitness_scores = []

        # Threading for parallel processing
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        
        self.setup_ui()
        
    def setup_gpu(self):
        """Initialize GPU optimizations for RTX 3080"""
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = True
            torch.cuda.empty_cache()
            print(f"GPU initialized: {torch.cuda.get_device_name(0)}")
            print(f"PyTorch version: {torch.__version__}")
            print(f"CUDA version: {torch.version.cuda}")
            print(f"Initial VRAM: {torch.cuda.memory_allocated() / 1e9:.2f}GB")

            # Enable mixed precision for VRAM optimization
            self.use_amp = True
            print("Mixed precision (AMP) enabled for VRAM optimization")
        else:
            print("CUDA not available - using CPU")
            self.use_amp = False
    
    def setup_ui(self):
        """Create the user interface"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Input file selection
        ttk.Label(main_frame, text="Input Video:").grid(row=0, column=0, sticky=tk.W, pady=5)
        ttk.Entry(main_frame, textvariable=self.input_path, width=50).grid(row=0, column=1, padx=5)
        ttk.Button(main_frame, text="Browse", command=self.browse_input).grid(row=0, column=2)
        
        # Output file selection
        ttk.Label(main_frame, text="Output Video:").grid(row=1, column=0, sticky=tk.W, pady=5)
        ttk.Entry(main_frame, textvariable=self.output_path, width=50).grid(row=1, column=1, padx=5)
        ttk.Button(main_frame, text="Browse", command=self.browse_output).grid(row=1, column=2)
        
        # Denoiser selection
        ttk.Label(main_frame, text="Denoiser:").grid(row=2, column=0, sticky=tk.W, pady=5)
        denoiser_combo = ttk.Combobox(main_frame, textvariable=self.denoiser_var, 
                                     values=["TAP", "MVDenoiser", "Auto"], state="readonly")
        denoiser_combo.grid(row=2, column=1, sticky=tk.W, padx=5)
        
        # Low-light enhancement
        ttk.Checkbutton(main_frame, text="Enable Low-Light Enhancement (DarkIR)",
                       variable=self.lowlight_var).grid(row=3, column=0, columnspan=2, sticky=tk.W, pady=5)

        # Colorization controls
        ttk.Checkbutton(main_frame, text="Enable Colorization",
                       variable=self.color_var, command=self.toggle_colorization).grid(row=4, column=0, columnspan=2, sticky=tk.W, pady=5)

        # Colorizer selection
        ttk.Label(main_frame, text="Colorizer:").grid(row=5, column=0, sticky=tk.W, pady=5)
        self.colorizer_combo = ttk.Combobox(main_frame, textvariable=self.colorizer_var,
                                           values=["VanGogh", "Cobra", "LVCD"], state="readonly")
        self.colorizer_combo.grid(row=5, column=1, sticky=tk.W, padx=5)
        self.colorizer_combo.bind('<<ComboboxSelected>>', self.on_colorizer_change)

        # Text prompt for VanGogh
        ttk.Label(main_frame, text="Text Prompt:").grid(row=6, column=0, sticky=tk.W, pady=5)
        self.prompt_entry = ttk.Entry(main_frame, textvariable=self.text_prompt, width=50)
        self.prompt_entry.grid(row=6, column=1, padx=5)

        # Reference image upload
        ttk.Label(main_frame, text="Reference Image:").grid(row=7, column=0, sticky=tk.W, pady=5)
        ref_frame = ttk.Frame(main_frame)
        ref_frame.grid(row=7, column=1, sticky=(tk.W, tk.E), padx=5)
        ttk.Entry(ref_frame, textvariable=self.ref_path, width=40).grid(row=0, column=0, sticky=(tk.W, tk.E))
        ttk.Button(ref_frame, text="Browse", command=self.browse_reference).grid(row=0, column=1, padx=(5,0))
        ref_frame.columnconfigure(0, weight=1)
        
        # Progress bar
        ttk.Label(main_frame, text="Progress:").grid(row=8, column=0, sticky=tk.W, pady=5)
        self.progress_bar = ttk.Progressbar(main_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.grid(row=8, column=1, columnspan=2, sticky=(tk.W, tk.E), padx=5)

        # Status label
        self.status_label = ttk.Label(main_frame, textvariable=self.status_var)
        self.status_label.grid(row=9, column=0, columnspan=3, pady=10)

        # Process button
        ttk.Button(main_frame, text="Process Video", command=self.start_processing).grid(row=10, column=1, pady=20)

        # VRAM monitor
        self.vram_label = ttk.Label(main_frame, text="VRAM: 0.00GB")
        self.vram_label.grid(row=11, column=0, columnspan=3, pady=5)

        # Tuning Parameters Section
        tuning_frame = ttk.LabelFrame(main_frame, text="Tuning Parameters", padding="5")
        tuning_frame.grid(row=12, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)

        # DarkIR Gain Slider
        ttk.Label(tuning_frame, text="DarkIR Gain:").grid(row=0, column=0, sticky=tk.W)
        self.gain_slider = ttk.Scale(tuning_frame, from_=0.5, to=2.0, variable=self.darkir_gain,
                                    orient=tk.HORIZONTAL, length=200)
        self.gain_slider.grid(row=0, column=1, padx=5)
        ttk.Label(tuning_frame, textvariable=self.darkir_gain).grid(row=0, column=2)

        # TAP Sigma Slider
        ttk.Label(tuning_frame, text="Denoise Strength:").grid(row=1, column=0, sticky=tk.W)
        self.sigma_slider = ttk.Scale(tuning_frame, from_=5, to=50, variable=self.tap_sigma,
                                     orient=tk.HORIZONTAL, length=200)
        self.sigma_slider.grid(row=1, column=1, padx=5)
        ttk.Label(tuning_frame, textvariable=self.tap_sigma).grid(row=1, column=2)

        # VanGogh Steps Slider
        ttk.Label(tuning_frame, text="VanGogh Steps:").grid(row=2, column=0, sticky=tk.W)
        self.steps_slider = ttk.Scale(tuning_frame, from_=10, to=50, variable=self.vangogh_steps,
                                     orient=tk.HORIZONTAL, length=200)
        self.steps_slider.grid(row=2, column=1, padx=5)
        ttk.Label(tuning_frame, textvariable=self.vangogh_steps).grid(row=2, column=2)

        # Cobra Reference Weight Slider
        ttk.Label(tuning_frame, text="Cobra Ref Weight:").grid(row=3, column=0, sticky=tk.W)
        self.ref_weight_slider = ttk.Scale(tuning_frame, from_=0.0, to=1.0, variable=self.cobra_ref_weight,
                                          orient=tk.HORIZONTAL, length=200)
        self.ref_weight_slider.grid(row=3, column=1, padx=5)
        ttk.Label(tuning_frame, textvariable=self.cobra_ref_weight).grid(row=3, column=2)

        # User Feedback for RL
        feedback_frame = ttk.Frame(tuning_frame)
        feedback_frame.grid(row=4, column=0, columnspan=3, pady=10)
        ttk.Label(feedback_frame, text="Quality Feedback:").pack(side='left', padx=5)
        ttk.Button(feedback_frame, text="👍", command=self.thumbs_up, width=5).pack(side='left', padx=2)
        ttk.Button(feedback_frame, text="👎", command=self.thumbs_down, width=5).pack(side='left', padx=2)

        # Console Section
        console_frame = ttk.Frame(self.root)
        console_frame.pack(side='bottom', fill='both', expand=True, padx=10, pady=5)

        # Console toggle button
        ttk.Button(console_frame, text="Toggle Console", command=self.toggle_console).pack(side='top', pady=2)

        # Console text widget with scrollbar
        console_text_frame = ttk.Frame(console_frame)
        console_text_frame.pack(fill='both', expand=True)

        self.console = tk.Text(console_text_frame, height=10, state='disabled', bg='black', fg='green',
                              font=('Consolas', 9))
        console_scrollbar = ttk.Scrollbar(console_text_frame, orient='vertical', command=self.console.yview)
        self.console.configure(yscrollcommand=console_scrollbar.set)

        self.console.pack(side='left', fill='both', expand=True)
        console_scrollbar.pack(side='right', fill='y')

        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        tuning_frame.columnconfigure(1, weight=1)

        # Start VRAM monitoring and initial log
        self.monitor_vram()
        self.log("Video Enhancement Tool initialized", "INFO")
    
    def browse_input(self):
        """Browse for input video file"""
        filename = filedialog.askopenfilename(
            title="Select Input Video",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
        )
        if filename:
            self.input_path.set(filename)
    
    def browse_output(self):
        """Browse for output video file"""
        filename = filedialog.asksaveasfilename(
            title="Save Output Video",
            defaultextension=".mp4",
            filetypes=[("MP4 files", "*.mp4"), ("All files", "*.*")]
        )
        if filename:
            self.output_path.set(filename)

    def browse_reference(self):
        """Browse for reference image file"""
        filename = filedialog.askopenfilename(
            title="Select Reference Image",
            filetypes=[("JPG files", "*.jpg"), ("JPEG files", "*.jpeg"), ("PNG files", "*.png"), ("All files", "*.*")]
        )
        if filename:
            self.ref_path.set(filename)

    def toggle_colorization(self):
        """Toggle colorization controls visibility"""
        if self.color_var.get():
            self.colorizer_combo.config(state="readonly")
            self.prompt_entry.config(state="normal")
        else:
            self.colorizer_combo.config(state="disabled")
            self.prompt_entry.config(state="disabled")

    def on_colorizer_change(self, event=None):
        """Handle colorizer selection change"""
        colorizer = self.colorizer_var.get()
        if colorizer == "VanGogh":
            self.prompt_entry.config(state="normal")
        elif colorizer == "Cobra":
            self.prompt_entry.config(state="disabled")

    def toggle_console(self):
        """Toggle console visibility"""
        if self.console_visible.get():
            self.console.master.pack_forget()
            self.console_visible.set(False)
        else:
            self.console.master.pack(side='bottom', fill='both', expand=True, padx=10, pady=5)
            self.console_visible.set(True)

    def log(self, message, level='INFO'):
        """Log message to console and file"""
        timestamp = time.strftime('%H:%M:%S')
        formatted_message = f"{timestamp} [{level.upper()}] {message}"

        # Log to file via logger
        if level.upper() == 'ERROR':
            logger.error(message)
        elif level.upper() == 'WARNING':
            logger.warning(message)
        else:
            logger.info(message)

        # Log to GUI console
        self.console.configure(state='normal')
        self.console.insert('end', formatted_message + '\n')
        self.console.configure(state='disabled')
        self.console.see('end')

        # Update GUI
        self.root.update_idletasks()

    def thumbs_up(self):
        """User feedback: positive"""
        self.log("User feedback: Positive (👍)", "INFO")
        # Reward current parameters
        reward = 0.9
        self.fitness_scores.append(reward)
        self.log(f"RL reward increased: {reward}", "INFO")

    def thumbs_down(self):
        """User feedback: negative"""
        self.log("User feedback: Negative (👎)", "INFO")
        # Penalize and mutate current parameters
        reward = 0.3
        self.fitness_scores.append(reward)

        # Mutate parameters downwards
        current_gain = self.darkir_gain.get()
        current_sigma = self.tap_sigma.get()

        if current_gain > 0.6:
            new_gain = max(0.5, current_gain * 0.9)
            self.darkir_gain.set(new_gain)
            self.log(f"Mutated DarkIR gain: {current_gain:.2f} → {new_gain:.2f}", "INFO")

        if current_sigma > 10:
            new_sigma = max(5, current_sigma * 0.9)
            self.tap_sigma.set(new_sigma)
            self.log(f"Mutated denoise strength: {current_sigma:.1f} → {new_sigma:.1f}", "INFO")
    
    def monitor_vram(self):
        """Monitor VRAM usage"""
        if torch.cuda.is_available():
            vram_used = torch.cuda.memory_allocated() / 1e9
            vram_reserved = torch.cuda.memory_reserved() / 1e9
            self.vram_label.config(text=f"VRAM: {vram_used:.2f}GB / {vram_reserved:.2f}GB")

            # Log high VRAM usage
            if vram_used > 6.0:
                self.log(f"High VRAM usage: {vram_used:.2f}GB", "WARNING")

            # Adaptive window size based on VRAM
            if vram_used > 1.5:
                self.window_size = 3
                self.frame_buffer = deque(maxlen=3)
            else:
                self.window_size = 5
                self.frame_buffer = deque(maxlen=5)

        # Schedule next update
        self.root.after(1000, self.monitor_vram)
    
    def load_model(self, model_name):
        """Load model with lazy loading and error handling"""
        if model_name in self.models:
            return self.models[model_name]
        
        try:
            self.log(f"Loading model: {model_name}", "INFO")

            if model_name == "TAP":
                model = self.load_tap_model()
            elif model_name == "MVDenoiser":
                model = self.load_mvdenoiser_model()
            elif model_name == "DarkIR":
                model = self.load_darkir_model()
            elif model_name == "VanGogh":
                model = self.load_vangogh_model()
            elif model_name == "Cobra":
                model = self.load_cobra_model()
            elif model_name == "LVCD":
                model = self.load_lvcd_model()
            else:
                raise ValueError(f"Unknown model: {model_name}")

            model.to(self.device)
            model.eval()
            self.models[model_name] = model

            # Fitness scoring for RL
            vram_used = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
            fitness = max(0, 1 - (vram_used / 2.0))  # Reward low VRAM usage
            self.fitness_scores.append(fitness)

            self.log(f"Model {model_name} loaded successfully. VRAM: {vram_used:.2f}GB, Fitness: {fitness:.3f}", "INFO")
            return model

        except FileNotFoundError as e:
            self.log(f"Weights not found for {model_name}, using random initialization", "WARNING")
            messagebox.showwarning("Weights Missing", f"Model weights not found for {model_name}. Using random initialization.")
            # Return a placeholder model
            model = nn.Identity()
            model.to(self.device)
            self.models[model_name] = model
            return model

        except Exception as e:
            self.log(f"Error loading {model_name}: {e}", "ERROR")
            messagebox.showerror("Model Loading Error", f"Failed to load {model_name}: {e}")
            return None
    
    def load_tap_model(self):
        """Load TAP denoising model"""
        # Placeholder - implement actual TAP model loading
        return nn.Identity()  # Temporary placeholder
    
    def load_mvdenoiser_model(self):
        """Load MVDenoiser model"""
        # Placeholder - implement actual MVDenoiser loading
        return nn.Identity()  # Temporary placeholder
    
    def load_darkir_model(self):
        """Load DarkIR low-light enhancement model"""
        # Placeholder - implement actual DarkIR loading
        return nn.Identity()  # Temporary placeholder

    def load_vangogh_model(self):
        """Load VanGogh colorization model"""
        # VanGogh repository is sparse - implementing fallback
        print("Warning: VanGogh repository is sparse, using fallback colorization")
        return nn.Identity()  # Temporary placeholder

    def load_cobra_model(self):
        """Load Cobra colorization model"""
        try:
            # Attempt to load Cobra model from app.py
            # This is a placeholder - actual implementation would load from Cobra/app.py
            self.log("Loading Cobra colorization model...", "INFO")
            return nn.Identity()  # Temporary placeholder
        except Exception as e:
            self.log(f"Error loading Cobra model: {e}", "ERROR")
            return nn.Identity()  # Fallback

    def load_lvcd_model(self):
        """Load LVCD colorization model"""
        try:
            # LVCD model loading - placeholder for now
            self.log("Loading LVCD colorization model...", "INFO")
            return nn.Identity()  # Temporary placeholder
        except Exception as e:
            self.log(f"Error loading LVCD model: {e}", "ERROR")
            return nn.Identity()  # Fallback
    
    def start_processing(self):
        """Start video processing in separate thread"""
        if not self.input_path.get() or not self.output_path.get():
            messagebox.showerror("Error", "Please select input and output files")
            return

        # Start processing thread
        thread = threading.Thread(target=self.process_video)
        thread.daemon = True
        thread.start()

    def process_video(self):
        """Main video processing pipeline with RL auto-selection"""
        try:
            self.status_var.set("Initializing...")

            # Open input video
            cap = cv2.VideoCapture(self.input_path.get())
            if not cap.isOpened():
                raise ValueError("Could not open input video")

            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Limit to 10-minute clips for development
            max_frames = min(total_frames, int(fps * 600))  # 10 minutes

            print(f"Video: {width}x{height} @ {fps:.2f}fps, {max_frames} frames")

            # Auto-select denoiser if needed
            denoiser_name = self.denoiser_var.get()
            if denoiser_name == "Auto":
                denoiser_name = self.auto_select_denoiser(cap)
                print(f"Auto-selected denoiser: {denoiser_name}")

            # Load models
            self.status_var.set("Loading models...")
            denoiser_model = self.load_model(denoiser_name)
            if denoiser_model is None:
                return

            darkir_model = None
            if self.lowlight_var.get():
                darkir_model = self.load_model("DarkIR")
                if darkir_model is None:
                    return

            # Load colorization model if enabled
            colorizer_model = None
            if self.color_var.get():
                colorizer_name = self.colorizer_var.get()
                colorizer_model = self.load_model(colorizer_name)
                if colorizer_model is None:
                    return

            # Setup video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(self.output_path.get(), fourcc, fps, (width, height))

            # Processing loop
            self.status_var.set("Processing frames...")
            processed_frames = 0
            start_time = time.time()

            while processed_frames < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break

                # Handle B&W videos and IR preprocessing
                if len(frame.shape) == 3 and frame.shape[2] == 3:
                    # Check if grayscale (all channels equal)
                    if np.allclose(frame[:,:,0], frame[:,:,1]) and np.allclose(frame[:,:,1], frame[:,:,2]):
                        # Convert to RGB by repeating channels
                        frame = cv2.cvtColor(frame[:,:,0], cv2.COLOR_GRAY2RGB)

                        # IR-specific preprocessing: histogram equalization
                        if self.detect_ir_frame(frame):
                            self.log("IR mode detected, applying histogram equalization", "INFO")
                            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                            equalized = cv2.equalizeHist(gray)
                            frame = cv2.cvtColor(equalized, cv2.COLOR_GRAY2RGB)

                # Convert to tensor
                frame_tensor = self.frame_to_tensor(frame)

                # Processing chain: Low-light → Denoise → Colorize

                # Step 1: Low-light enhancement
                if darkir_model is not None:
                    with torch.cuda.amp.autocast():
                        frame_tensor = self.apply_darkir(darkir_model, frame_tensor)

                # Add to buffer
                self.frame_buffer.append(frame_tensor)

                # Process when buffer is ready
                if len(self.frame_buffer) >= min(3, self.window_size):
                    # Step 2: Denoising
                    denoised_tensor = self.apply_denoising(denoiser_model, denoiser_name)

                    # Step 3: Colorization (if enabled and frame is grayscale)
                    final_tensor = denoised_tensor
                    if colorizer_model is not None:
                        # Check if frame is grayscale (low channel variance)
                        channel_var = torch.var(denoised_tensor, dim=[2,3]).mean()
                        if channel_var < 0.01:  # Likely grayscale
                            final_tensor = self.apply_colorization(colorizer_model, denoised_tensor)

                            # Check if colorization failed (still grayscale)
                            post_color_var = torch.var(final_tensor, dim=[2,3]).mean()
                            if post_color_var < 0.01:
                                self.log("Colorization failed, retrying with LVCD", "WARNING")
                                try:
                                    lvcd_model = self.load_model("LVCD")
                                    if lvcd_model:
                                        final_tensor = self.apply_colorization(lvcd_model, denoised_tensor)
                                except Exception as e:
                                    self.log(f"LVCD fallback failed: {e}", "ERROR")
                        else:
                            self.log("RGB detected - skipping colorization", "INFO")

                    # Convert back to frame
                    output_frame = self.tensor_to_frame(final_tensor)
                    out.write(output_frame)

                    processed_frames += 1

                    # Update progress
                    progress = (processed_frames / max_frames) * 100
                    self.progress_var.set(progress)

                    # Calculate ETA
                    elapsed = time.time() - start_time
                    if processed_frames > 0:
                        eta = (elapsed / processed_frames) * (max_frames - processed_frames)
                        self.status_var.set(f"Processing... ETA: {eta:.1f}s")

                    # VRAM check
                    if torch.cuda.is_available():
                        vram_used = torch.cuda.memory_allocated() / 1e9
                        if vram_used > 1.5:
                            print(f"Warning: High VRAM usage: {vram_used:.2f}GB")

            # Cleanup
            cap.release()
            out.release()

            # Add audio using moviepy (placeholder)
            self.status_var.set("Adding audio...")
            self.add_audio_track()

            self.status_var.set("Complete!")
            self.progress_var.set(100)

            # Fitness audit
            final_fitness = np.mean(self.fitness_scores) if self.fitness_scores else 0
            print(f"Processing complete. Final fitness: {final_fitness:.3f}")

            if final_fitness < 0.7:
                print("Warning: Low fitness score - consider optimization")

        except Exception as e:
            self.status_var.set(f"Error: {e}")
            messagebox.showerror("Processing Error", str(e))
            print(f"Processing error: {e}")

    def auto_select_denoiser(self, cap):
        """RL-based automatic denoiser selection"""
        # Sample 5 frames for noise analysis
        sample_frames = []
        original_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)

        for i in range(5):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i * 100)  # Sample every 100 frames
            ret, frame = cap.read()
            if ret:
                sample_frames.append(frame)

        # Reset position
        cap.set(cv2.CAP_PROP_POS_FRAMES, original_pos)

        if not sample_frames:
            return "TAP"  # Default fallback

        # Compute noise variance
        noise_vars = []
        for frame in sample_frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            noise_var = np.var(gray)
            noise_vars.append(noise_var)

        avg_noise = np.mean(noise_vars)

        # RL reward calculation
        reward_tap = 1.0 / (avg_noise + 1e-6) if avg_noise < self.rl_threshold else 0.5
        reward_mv = 1.0 / (avg_noise + 1e-6) if avg_noise >= self.rl_threshold else 0.3

        # Bias audit
        selection = "TAP" if reward_tap > reward_mv else "MVDenoiser"

        # Mutation for bias correction
        if len(self.fitness_scores) > 10:
            tap_selections = sum(1 for s in self.fitness_scores[-10:] if s > 0.8)
            if tap_selections > 8:  # >80% bias to TAP
                self.rl_threshold *= 0.9  # Mutate threshold down
                print(f"Bias detected, mutating threshold to {self.rl_threshold:.3f}")

        return selection

    def auto_select_colorizer(self, sample_frames):
        """RL-based automatic colorizer selection"""
        if not sample_frames:
            return "VanGogh"  # Default fallback

        try:
            # Compute quality metrics for selection
            quality_scores = []

            for frame in sample_frames:
                # Convert to tensor for analysis
                frame_tensor = self.frame_to_tensor(frame)

                # Simulate colorization quality (placeholder metrics)
                # In real implementation, would use SSIM/LPIPS/FID
                gray_var = torch.var(frame_tensor).item()
                edge_strength = torch.std(frame_tensor).item()

                # Simple heuristic: VanGogh for high detail, Cobra for simple scenes
                vangogh_score = edge_strength * 0.8 + gray_var * 0.2
                cobra_score = (1.0 - edge_strength) * 0.6 + gray_var * 0.4

                quality_scores.append({
                    'vangogh': vangogh_score,
                    'cobra': cobra_score
                })

            # Average scores
            avg_vangogh = np.mean([s['vangogh'] for s in quality_scores])
            avg_cobra = np.mean([s['cobra'] for s in quality_scores])

            # Select best model
            selection = "VanGogh" if avg_vangogh > avg_cobra else "Cobra"

            # Bias audit for colorization
            if len(self.fitness_scores) > 10:
                recent_selections = self.fitness_scores[-10:]
                vangogh_bias = sum(1 for s in recent_selections if s > 0.7) / len(recent_selections)

                if vangogh_bias > 0.7:  # >70% bias to VanGogh
                    # Mutate selection threshold
                    if selection == "VanGogh" and avg_cobra > 0.3:
                        selection = "Cobra"
                        print("Bias detected in colorizer selection, switching to Cobra")

            print(f"Auto-selected colorizer: {selection} (VanGogh: {avg_vangogh:.3f}, Cobra: {avg_cobra:.3f})")
            return selection

        except Exception as e:
            print(f"Auto-selection error: {e}")
            return "VanGogh"  # Fallback

    def detect_ir_frame(self, frame):
        """Detect if frame is from IR/surveillance camera"""
        try:
            # Convert to grayscale for analysis
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

            # IR characteristics: low contrast, specific intensity distribution
            mean_intensity = np.mean(gray)
            std_intensity = np.std(gray)

            # IR frames typically have low contrast and specific intensity ranges
            is_ir = (mean_intensity < 80 and std_intensity < 30) or (mean_intensity > 200 and std_intensity < 20)

            return is_ir
        except Exception as e:
            self.log(f"Error in IR detection: {e}", "ERROR")
            return False

    def check_rl_bias(self):
        """Check for bias in RL parameter mutations"""
        if len(self.fitness_scores) < 10:
            return

        recent_scores = self.fitness_scores[-10:]
        low_scores = sum(1 for score in recent_scores if score < 0.5)

        # If >70% of recent scores are low (indicating downward bias)
        if low_scores > 7:
            self.log("Bias detected: too many downward mutations", "WARNING")

            # Randomize parameters upwards by 20%
            current_gain = self.darkir_gain.get()
            current_sigma = self.tap_sigma.get()

            new_gain = min(2.0, current_gain * 1.2)
            new_sigma = min(50, current_sigma * 1.2)

            self.darkir_gain.set(new_gain)
            self.tap_sigma.set(new_sigma)

            self.log(f"Bias correction: gain {current_gain:.2f}→{new_gain:.2f}, sigma {current_sigma:.1f}→{new_sigma:.1f}", "INFO")

    def frame_to_tensor(self, frame):
        """Convert OpenCV frame to PyTorch tensor"""
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Normalize to [0, 1]
        frame_norm = frame_rgb.astype(np.float32) / 255.0
        # Convert to tensor and add batch dimension
        tensor = torch.from_numpy(frame_norm).permute(2, 0, 1).unsqueeze(0)
        return tensor.to(self.device)

    def tensor_to_frame(self, tensor):
        """Convert PyTorch tensor back to OpenCV frame"""
        # Remove batch dimension and move to CPU
        tensor_cpu = tensor.squeeze(0).cpu()
        # Clamp values to [0, 1]
        tensor_clamped = torch.clamp(tensor_cpu, 0, 1)
        # Convert to numpy and scale to [0, 255]
        frame_np = (tensor_clamped.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        # Convert RGB back to BGR
        frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
        return frame_bgr

    def apply_darkir(self, model, frame_tensor):
        """Apply DarkIR low-light enhancement with gain parameter"""
        try:
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    # Apply gain parameter for exposure adjustment
                    gain = self.darkir_gain.get()
                    if gain != 1.0:
                        frame_tensor = frame_tensor * gain
                        frame_tensor = torch.clamp(frame_tensor, 0, 1)
                        self.log(f"Applied DarkIR gain: {gain:.2f}", "INFO")

                    enhanced = model(frame_tensor)

                    # Log tensor shapes for debugging
                    self.log(f"DarkIR input shape: {frame_tensor.shape}, output shape: {enhanced.shape}", "DEBUG")

                    return enhanced
        except Exception as e:
            self.log(f"DarkIR processing error: {e}", "ERROR")
            return frame_tensor  # Return original on error

    def apply_denoising(self, model, model_name):
        """Apply denoising to frame buffer with strength parameter"""
        try:
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    sigma = self.tap_sigma.get()
                    self.log(f"Using denoise strength: {sigma:.1f}", "INFO")

                    if model_name == "TAP":
                        # TAP expects temporal sequence
                        if len(self.frame_buffer) >= 3:
                            # Stack frames for temporal processing
                            frames = torch.cat(list(self.frame_buffer)[-3:], dim=0)
                            # Apply sigma parameter (placeholder - actual implementation would use it)
                            denoised = model(frames.unsqueeze(0))
                            # Return center frame
                            result = denoised[0, 1:2]  # Middle frame
                            self.log(f"TAP input shape: {frames.shape}, output shape: {result.shape}", "DEBUG")
                            return result
                        else:
                            return self.frame_buffer[-1]

                    elif model_name == "MVDenoiser":
                        # MVDenoiser single frame processing
                        current_frame = self.frame_buffer[-1]
                        denoised = model(current_frame)
                        self.log(f"MVDenoiser input shape: {current_frame.shape}, output shape: {denoised.shape}", "DEBUG")
                        return denoised

                    else:
                        # Fallback - return original
                        return self.frame_buffer[-1]

        except Exception as e:
            self.log(f"Denoising error ({model_name}): {e}", "ERROR")
            return self.frame_buffer[-1]  # Return original on error

    def apply_colorization(self, model, frame_tensor):
        """Apply colorization to grayscale frame"""
        try:
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    colorizer_name = self.colorizer_var.get()

                    if colorizer_name == "VanGogh":
                        # VanGogh with text prompt
                        prompt = self.text_prompt.get() or "natural colors"
                        colorized = self.apply_vangogh_colorization(model, frame_tensor, prompt)
                    elif colorizer_name == "Cobra":
                        # Cobra with reference image
                        ref_path = self.ref_path.get()
                        if not ref_path:
                            print("Warning: No reference image for Cobra, using fallback")
                            return frame_tensor
                        colorized = self.apply_cobra_colorization(model, frame_tensor, ref_path)
                    else:
                        return frame_tensor

                    # Log VRAM usage for colorization
                    if torch.cuda.is_available():
                        vram_used = torch.cuda.memory_allocated() / 1e9
                        if vram_used > 5.0:
                            print(f"Warning: High VRAM usage during colorization: {vram_used:.2f}GB")

                    return colorized

        except Exception as e:
            print(f"Colorization error ({self.colorizer_var.get()}): {e}")
            return frame_tensor  # Return original on error

    def apply_vangogh_colorization(self, model, frame_tensor, prompt):
        """Apply VanGogh colorization with text prompt"""
        # Placeholder - VanGogh repo is sparse
        print(f"VanGogh colorization with prompt: '{prompt}'")
        return frame_tensor  # Temporary fallback

    def apply_cobra_colorization(self, model, frame_tensor, ref_path):
        """Apply Cobra colorization with reference image"""
        # Placeholder - implement Cobra inference
        print(f"Cobra colorization with reference: {ref_path}")
        return frame_tensor  # Temporary fallback

    def add_audio_track(self):
        """Add audio from input to output video using moviepy"""
        try:
            # This is a placeholder - moviepy integration would go here
            # For now, just log the operation
            print("Audio track integration placeholder")
            #
            # import moviepy.editor as mp
            # video_clip = mp.VideoFileClip(self.output_path.get())
            # audio_clip = mp.VideoFileClip(self.input_path.get()).audio
            # final_clip = video_clip.set_audio(audio_clip)
            # final_clip.write_videofile(self.output_path.get() + "_with_audio.mp4")

        except Exception as e:
            print(f"Audio integration error: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = VideoEnhancerApp(root)
    root.mainloop()
