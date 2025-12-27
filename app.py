"""
DES COLOR IMAGE ENCRYPTION TOOL
All-in-one dengan UI rapi dan key visible
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk, scrolledtext
from PIL import Image, ImageTk
import numpy as np
import time
import os

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from Crypto.Cipher import DES
from Crypto.Util.Padding import pad, unpad

# ================= DES ENCRYPTION FUNCTIONS =================
def encrypt_image_color(image_array: np.ndarray, key: bytes) -> np.ndarray:
    """Encrypt a color image (RGB) using DES in ECB mode"""
    if len(image_array.shape) != 3 or image_array.shape[2] != 3:
        raise ValueError("Input must be an RGB image with shape (h, w, 3)")
    
    height, width, channels = image_array.shape
    
    # Flatten the image to 1D array
    flat_data = image_array.flatten().tobytes()
    
    # Create DES cipher
    cipher = DES.new(key, DES.MODE_ECB)
    
    # Pad data to multiple of block size
    padded_data = pad(flat_data, 8)
    
    # Encrypt
    encrypted_bytes = cipher.encrypt(padded_data)
    
    # Convert back to numpy array
    encrypted_flat = np.frombuffer(encrypted_bytes[:len(flat_data)], dtype=np.uint8)
    
    # Reshape back to original image shape
    encrypted_array = encrypted_flat.reshape(height, width, channels)
    
    return encrypted_array

def decrypt_image_color(image_array: np.ndarray, key: bytes) -> np.ndarray:
    """Decrypt a color image (RGB) using DES in ECB mode"""
    if len(image_array.shape) != 3 or image_array.shape[2] != 3:
        raise ValueError("Input must be an RGB image with shape (h, w, 3)")
    
    height, width, channels = image_array.shape
    
    # Flatten the encrypted image
    flat_data = image_array.flatten().tobytes()
    
    # Create DES cipher
    cipher = DES.new(key, DES.MODE_ECB)
    
    # PERBAIKAN: Encrypt dulu untuk mendapatkan panjang data terpadded
    # Hitung berapa banyak padding yang ditambahkan saat enkripsi
    original_len = height * width * channels
    padded_len = ((original_len + 7) // 8) * 8  # Round up to nearest multiple of 8
    
    # Jika data sudah sesuai dengan panjang terpadded, gunakan langsung
    if len(flat_data) == padded_len:
        padded_encrypted = flat_data
    else:
        # Tambahkan padding jika perlu
        padded_encrypted = pad(flat_data, 8)
    
    # Decrypt
    decrypted_bytes = cipher.decrypt(padded_encrypted)
    
    # PERBAIKAN: Pastikan kita mengambil jumlah byte yang tepat
    decrypted_bytes = decrypted_bytes[:original_len]
    
    # Convert back to numpy array
    decrypted_flat = np.frombuffer(decrypted_bytes, dtype=np.uint8)
    
    # Reshape back to original image shape
    decrypted_array = decrypted_flat.reshape(height, width, channels)
    
    return decrypted_array

# ================= METRICS FUNCTIONS =================
def calculate_entropy(img):
    """Calculate Shannon entropy for color or grayscale images"""
    if len(img.shape) == 2:  # Grayscale
        hist, _ = np.histogram(img.flatten(), bins=256, range=(0, 256))
    else:  # Color (RGB)
        # Calculate entropy for each channel and average
        entropies = []
        for channel in range(3):
            channel_data = img[:, :, channel].flatten()
            hist, _ = np.histogram(channel_data, bins=256, range=(0, 256))
            prob = hist / np.sum(hist)
            prob_nonzero = prob[prob > 0]
            entropy_ch = -np.sum(prob_nonzero * np.log2(prob_nonzero))
            entropies.append(entropy_ch)
        return np.mean(entropies)
    
    prob = hist / np.sum(hist)
    prob_nonzero = prob[prob > 0]
    return -np.sum(prob_nonzero * np.log2(prob_nonzero))

def calculate_psnr(img1, img2):
    """Calculate PSNR for color images"""
    if img1.shape != img2.shape:
        img2 = np.array(Image.fromarray(img2).resize((img1.shape[1], img1.shape[0])))
    
    if len(img1.shape) == 3:  # Color
        mse_channels = []
        for i in range(3):
            mse = np.mean((img1[:,:,i].astype(float) - img2[:,:,i].astype(float)) ** 2)
            mse_channels.append(mse)
        mse = np.mean(mse_channels)
    else:  # Grayscale
        mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
    
    if mse == 0:
        return float("inf")
    return 20 * np.log10(255.0 / np.sqrt(mse))

def calculate_npcr(img1, img2):
    """Calculate NPCR for color images"""
    if img1.shape != img2.shape:
        raise ValueError("Images must have same dimensions")
    
    if len(img1.shape) == 3:  # Color
        # Check if ALL THREE channels are different (not just any one)
        # Only count as "different" if R AND G AND B are all different
        diff = np.logical_and(np.logical_and(img1[:,:,0] != img2[:,:,0],
                                            img1[:,:,1] != img2[:,:,1]),
                             img1[:,:,2] != img2[:,:,2])
    else:  # Grayscale
        diff = img1 != img2
    
    return np.sum(diff) / (img1.shape[0] * img1.shape[1]) * 100

def calculate_uaci(img1, img2):
    """Calculate UACI for color images"""
    if img1.shape != img2.shape:
        raise ValueError("Images must have same dimensions")
    
    if len(img1.shape) == 3:  # Color
        diff_sum = 0
        total_pixels = img1.shape[0] * img1.shape[1] * 3
        for i in range(3):
            diff_sum += np.sum(np.abs(img1[:,:,i].astype(float) - img2[:,:,i].astype(float)))
        return diff_sum / (total_pixels * 255.0) * 100
    else:  # Grayscale
        diff = np.abs(img1.astype(float) - img2.astype(float))
        return np.sum(diff) / (img1.size * 255.0) * 100

# ================= GLOBAL VARIABLES =================
selected_image_path = None
canvas_widget = None
original_img_rgb = None
encrypted_img_rgb = None
decrypted_img_rgb = None
analysis_log = []

# ================= LOGGING FUNCTION =================
def log_message(message):
    """Add message to log and console"""
    timestamp = time.strftime("%H:%M:%S", time.localtime())
    log_entry = f"[{timestamp}] {message}"
    analysis_log.append(log_entry)
    print(log_entry)
    
    # Update log display if widget exists
    if 'log_text' in globals():
        log_text.insert(tk.END, log_entry + "\n")
        log_text.see(tk.END)

# ================= GUI FUNCTIONS =================
def upload_image():
    global selected_image_path, original_img_rgb
    path = filedialog.askopenfilename(
        title="Select Image File",
        filetypes=[
            ("Image Files", "*.png *.jpg *.jpeg *.bmp"),
            ("PNG files", "*.png"),
            ("JPEG files", "*.jpg *.jpeg"),
            ("Bitmap files", "*.bmp"),
            ("All files", "*.*")
        ]
    )
    
    if path:
        selected_image_path = path
        filename = os.path.basename(path)
        file_size = os.path.getsize(path) / 1024  # Size in KB
        
        label_path.config(text=f"📁 {filename} ({file_size:.1f} KB)")
        log_message(f"Image loaded: {filename}")
        
        try:
            # Load and preview original image
            original_img_rgb = Image.open(path)
            
            # Get image info
            width, height = original_img_rgb.size
            mode = original_img_rgb.mode
            
            # Update info labels
            info_original.config(text=f"Size: {width}×{height} | Mode: {mode}")
            
            # Show preview
            preview_img = original_img_rgb.copy()
            preview_img.thumbnail((180, 180), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(preview_img)
            preview_label.config(image=photo)
            preview_label.image = photo
            preview_label.config(text="")
            
            # Enable process button
            process_button.config(state="normal")
            
            log_message(f"Image info: {width}x{height} pixels, {mode} mode")
            
        except Exception as e:
            messagebox.showerror("Load Error", f"Cannot load image:\n{str(e)}")
            preview_label.config(text="❌ Load failed", image='')

def toggle_key_visibility():
    """Toggle between showing and hiding the key"""
    if entry_key.cget('show') == '*':
        entry_key.config(show='')
        btn_toggle_key.config(text="🙈 Hide")
    else:
        entry_key.config(show='*')
        btn_toggle_key.config(text="👁 Show")

def validate_key():
    """Validate the encryption key"""
    key = entry_key.get()
    if len(key) == 8:
        key_status.config(text="✓ Valid (8 chars)", fg="green")
        return True
    else:
        key_status.config(text=f"✗ Need {8-len(key)} more chars", fg="red")
        return False

def process_image():
    global selected_image_path, original_img_rgb, encrypted_img_rgb, decrypted_img_rgb
    
    if not selected_image_path:
        messagebox.showwarning("No Image", "Please upload an image first!")
        return
    
    key = entry_key.get()
    if len(key) != 8:
        messagebox.showerror("Invalid Key", "DES key must be exactly 8 characters!")
        entry_key.focus_set()
        return
    
    # Clear previous results
    for widget in results_frame.winfo_children():
        widget.destroy()
    
    # Show processing indicator
    progress_bar.start()
    process_button.config(state="disabled", text="Processing...")
    status_label.config(text="Encrypting and analyzing...")
    root.update()
    
    try:
        log_message("=" * 60)
        log_message("STARTING IMAGE ENCRYPTION PROCESS")
        log_message("=" * 60)
        
        # Load original image
        img_rgb = Image.open(selected_image_path).convert("RGB")
        orig_array = np.array(img_rgb)
        
        height, width, _ = orig_array.shape
        log_message(f"Processing image: {width}x{height} pixels (RGB)")
        
        # Encrypt the image
        log_message(f"Encrypting with key: '{key}'")
        start_time = time.time()
        encrypted_array = encrypt_image_color(orig_array, key.encode())
        encryption_time = time.time() - start_time
        log_message(f"Encryption completed in {encryption_time:.3f} seconds")
        
        # Decrypt the image
        log_message("Decrypting image...")
        start_time = time.time()
        decrypted_array = decrypt_image_color(encrypted_array, key.encode())
        decryption_time = time.time() - start_time
        log_message(f"Decryption completed in {decryption_time:.3f} seconds")
        
        # Convert back to PIL Images
        encrypted_img_rgb = Image.fromarray(encrypted_array.astype('uint8'))
        decrypted_img_rgb = Image.fromarray(decrypted_array.astype('uint8'))
        
        # Calculate metrics
        log_message("Calculating metrics...")
        
        # Prepare metrics data
        metrics_data = {
            "Original": {
                "Entropy": calculate_entropy(orig_array),
                "Size": f"{width}×{height}",
                "Channels": "RGB",
                "PSNR": "-",
                "NPCR": "-",
                "UACI": "-"
            },
            "Encrypted": {
                "Entropy": calculate_entropy(encrypted_array),
                "Encryption Time": f"{encryption_time:.3f}s",
                "PSNR": calculate_psnr(orig_array, encrypted_array),
                "NPCR": calculate_npcr(orig_array, encrypted_array),
                "UACI": calculate_uaci(orig_array, encrypted_array)
            },
            "Decrypted": {
                "Entropy": calculate_entropy(decrypted_array),
                "Decryption Time": f"{decryption_time:.3f}s",
                "PSNR": calculate_psnr(orig_array, decrypted_array),
                "NPCR": calculate_npcr(orig_array, decrypted_array),
                "UACI": calculate_uaci(orig_array, decrypted_array)
            }
        }
        
        # Display results in a table
        display_results_table(metrics_data)
        
        # Create visualization with histograms only
        create_histogram_visualization(orig_array, encrypted_array, decrypted_array, metrics_data)
        
        # Enable save buttons
        btn_save_encrypted.config(state="normal")
        btn_save_decrypted.config(state="normal")
        
        # Log summary
        log_message("\n" + "=" * 60)
        log_message("ENCRYPTION ANALYSIS SUMMARY")
        log_message("=" * 60)
        log_message(f"Original Entropy: {metrics_data['Original']['Entropy']:.4f}")
        log_message(f"Encrypted Entropy: {metrics_data['Encrypted']['Entropy']:.4f}")
        log_message(f"Decrypted Entropy: {metrics_data['Decrypted']['Entropy']:.4f}")
        log_message(f"NPCR: {metrics_data['Encrypted']['NPCR']:.4f}%")
        log_message(f"UACI: {metrics_data['Encrypted']['UACI']:.4f}%")
        log_message(f"PSNR (Decrypted): {metrics_data['Decrypted']['PSNR']:.2f} dB")
        
        # Quality assessment
        npcr_val = metrics_data['Encrypted']['NPCR']
        uaci_val = metrics_data['Encrypted']['UACI']
        psnr_val = metrics_data['Decrypted']['PSNR']
        
        if npcr_val > 99.5 and uaci_val > 33.0:
            encryption_quality = "EXCELLENT"
        elif npcr_val > 99.0 and uaci_val > 32.0:
            encryption_quality = "GOOD"
        else:
            encryption_quality = "FAIR"
        
        if psnr_val == float('inf'):
            decryption_quality = "PERFECT (Lossless)"
        elif psnr_val > 40:
            decryption_quality = "EXCELLENT"
        elif psnr_val > 30:
            decryption_quality = "GOOD"
        else:
            decryption_quality = "POOR"
        
        log_message(f"\nEncryption Quality: {encryption_quality}")
        log_message(f"Decryption Quality: {decryption_quality}")
        log_message("=" * 60)
        
        status_label.config(text=f"Analysis complete! Encryption: {encryption_quality}, Decryption: {decryption_quality}")
        
    except Exception as e:
        error_msg = f"Error during processing:\n{str(e)}"
        messagebox.showerror("Processing Error", error_msg)
        log_message(f"ERROR: {error_msg}")
        status_label.config(text="Error occurred during processing")
        
    finally:
        # Reset UI
        progress_bar.stop()
        process_button.config(state="normal", text="🔐 Encrypt & Analyze")

def display_results_table(metrics_data):
    """Display metrics in a nice table format"""
    # Create a frame for the table
    table_frame = ttk.Frame(results_frame)
    table_frame.pack(fill="both", expand=True, padx=10, pady=10)
    
    # Table headers
    headers = ["Metric", "Original", "Encrypted", "Decrypted"]
    
    # Create table
    for col, header in enumerate(headers):
        header_label = tk.Label(table_frame, text=header, font=("Arial", 10, "bold"),
                              bg="#f0f0f0", relief="ridge", padx=10, pady=5)
        header_label.grid(row=0, column=col, sticky="nsew")
    
    # Define the metrics to display
    display_metrics = [
        ("Entropy", "{:.4f}", "{:.4f}", "{:.4f}"),
        ("Size/Time", "{}", "{} s", "{} s"),
        ("Channels/PSNR", "{}", "{:.2f} dB", "{:.2f} dB"),
        ("NPCR", "{}", "{:.4f}%", "{:.4f}%"),
        ("UACI", "{}", "{:.4f}%", "{:.4f}%")
    ]
    
    # Fill the table
    for row, (metric_name, orig_fmt, enc_fmt, dec_fmt) in enumerate(display_metrics, 1):
        # Metric name
        metric_label = tk.Label(table_frame, text=metric_name, font=("Arial", 9),
                              bg="#f8f8f8", relief="ridge", padx=10, pady=5)
        metric_label.grid(row=row, column=0, sticky="nsew")
        
        # Original value
        if metric_name == "Entropy":
            value = metrics_data["Original"]["Entropy"]
            text = orig_fmt.format(value)
        elif metric_name == "Size/Time":
            text = metrics_data["Original"]["Size"]
        elif metric_name == "Channels/PSNR":
            text = metrics_data["Original"]["Channels"]
        elif metric_name == "NPCR":
            text = metrics_data["Original"]["NPCR"]
        elif metric_name == "UACI":
            text = metrics_data["Original"]["UACI"]
        
        orig_label = tk.Label(table_frame, text=text, font=("Courier", 9),
                            bg="white", relief="ridge", padx=10, pady=5)
        orig_label.grid(row=row, column=1, sticky="nsew")
        
        # Encrypted value
        if metric_name == "Entropy":
            value = metrics_data["Encrypted"]["Entropy"]
            text = enc_fmt.format(value)
        elif metric_name == "Size/Time":
            text = metrics_data["Encrypted"]["Encryption Time"]
        elif metric_name == "Channels/PSNR":
            value = metrics_data["Encrypted"]["PSNR"]
            text = enc_fmt.format(value)
        elif metric_name == "NPCR":
            value = metrics_data["Encrypted"]["NPCR"]
            text = enc_fmt.format(value)
        elif metric_name == "UACI":
            value = metrics_data["Encrypted"]["UACI"]
            text = enc_fmt.format(value)
        
        enc_label = tk.Label(table_frame, text=text, font=("Courier", 9),
                           bg="#fff0f0", relief="ridge", padx=10, pady=5)
        enc_label.grid(row=row, column=2, sticky="nsew")
        
        # Decrypted value
        if metric_name == "Entropy":
            value = metrics_data["Decrypted"]["Entropy"]
            text = dec_fmt.format(value)
        elif metric_name == "Size/Time":
            text = metrics_data["Decrypted"]["Decryption Time"]
        elif metric_name == "Channels/PSNR":
            value = metrics_data["Decrypted"]["PSNR"]
            text = dec_fmt.format(value)
        elif metric_name == "NPCR":
            value = metrics_data["Decrypted"]["NPCR"]
            text = dec_fmt.format(value)
        elif metric_name == "UACI":
            value = metrics_data["Decrypted"]["UACI"]
            text = dec_fmt.format(value)
        
        dec_label = tk.Label(table_frame, text=text, font=("Courier", 9),
                           bg="#f0fff0", relief="ridge", padx=10, pady=5)
        dec_label.grid(row=row, column=3, sticky="nsew")
    
    # Configure grid weights
    for i in range(4):
        table_frame.columnconfigure(i, weight=1)

def create_histogram_visualization(orig, enc, dec, metrics):
    """Create visualization with histograms only"""
    global canvas_widget
    
    if canvas_widget:
        canvas_widget.get_tk_widget().destroy()
    
    fig = Figure(figsize=(14, 10), dpi=90)
    
    # Create a 3x2 grid
    # Row 0: Images
    # Row 1: Histograms
    # Row 2: Combined histogram comparison
    
    # Row 0: Images
    titles = ["Original Image", "Encrypted Image", "Decrypted Image"]
    images = [orig, enc, dec]
    
    for i in range(3):
        ax = fig.add_subplot(3, 3, i + 1)
        ax.imshow(images[i])
        ax.set_title(titles[i], fontsize=11, fontweight='bold', pad=10)
        ax.set_xlabel(f"{images[i].shape[1]}×{images[i].shape[0]}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Row 1: Individual Histograms
    hist_titles = ["Original Histogram", "Encrypted Histogram", "Decrypted Histogram"]
    hist_colors = ['#3498db', '#e74c3c', '#2ecc71']  # Blue, Red, Green
    
    for i in range(3):
        ax = fig.add_subplot(3, 3, i + 4)
        
        # Convert image to grayscale for histogram
        if len(images[i].shape) == 3:  # RGB image
            gray_img = np.mean(images[i], axis=2).flatten()
        else:  # Already grayscale
            gray_img = images[i].flatten()
        
        # Calculate histogram
        hist, bins = np.histogram(gray_img, bins=50, range=(0, 256))
        bin_centers = (bins[:-1] + bins[1:]) / 2
        
        # Plot histogram
        ax.bar(bin_centers, hist, width=4.8, color=hist_colors[i], alpha=0.7, edgecolor='black', linewidth=0.5)
        
        ax.set_title(hist_titles[i], fontsize=11, fontweight='bold', pad=10)
        ax.set_xlabel("Pixel Intensity (0-255)", fontsize=9)
        ax.set_ylabel("Frequency", fontsize=9)
        ax.grid(True, alpha=0.3, linestyle='--', axis='y')
        ax.set_xlim([0, 255])
        
        # Calculate and display statistics
        mean_val = np.mean(gray_img)
        std_val = np.std(gray_img)
        entropy_val = metrics['Original']['Entropy'] if i == 0 else metrics['Encrypted']['Entropy'] if i == 1 else metrics['Decrypted']['Entropy']
        
        stats_text = f"Mean: {mean_val:.1f}\nStd: {std_val:.1f}\nEntropy: {entropy_val:.4f}"
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=8,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Row 2: Combined Histogram Comparison (spanning all 3 columns)
    ax_combined = fig.add_subplot(3, 1, 3)
    
    # Prepare data for combined histogram
    gray_data = []
    labels = ["Original", "Encrypted", "Decrypted"]
    
    for img in images:
        if len(img.shape) == 3:
            gray_data.append(np.mean(img, axis=2).flatten())
        else:
            gray_data.append(img.flatten())
    
    # Create histogram for each image
    for i, data in enumerate(gray_data):
        hist, bins = np.histogram(data, bins=50, range=(0, 256), density=True)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        ax_combined.plot(bin_centers, hist, linewidth=2, label=labels[i], color=hist_colors[i])
    
    ax_combined.set_title("Combined Histogram Comparison (Normalized)", fontsize=12, fontweight='bold', pad=10)
    ax_combined.set_xlabel("Pixel Intensity (0-255)", fontsize=10)
    ax_combined.set_ylabel("Normalized Frequency", fontsize=10)
    ax_combined.grid(True, alpha=0.3, linestyle='--')
    ax_combined.legend(loc='upper right', fontsize=9)
    ax_combined.set_xlim([0, 255])
    
   
    
    fig.tight_layout(pad=3.0)
    
    canvas_widget = FigureCanvasTkAgg(fig, master=viz_frame)
    canvas_widget.draw()
    canvas_widget.get_tk_widget().pack(fill="both", expand=True)

def save_encrypted_image():
    if encrypted_img_rgb is not None:
        file_path = filedialog.asksaveasfilename(
            title="Save Encrypted Image",
            defaultextension=".png",
            filetypes=[
                ("PNG files", "*.png"),
                ("JPEG files", "*.jpg"),
                ("All files", "*.*")
            ]
        )
        if file_path:
            encrypted_img_rgb.save(file_path)
            messagebox.showinfo("Saved", f"Encrypted image saved to:\n{file_path}")
            log_message(f"Encrypted image saved: {os.path.basename(file_path)}")

def save_decrypted_image():
    if decrypted_img_rgb is not None:
        file_path = filedialog.asksaveasfilename(
            title="Save Decrypted Image",
            defaultextension=".png",
            filetypes=[
                ("PNG files", "*.png"),
                ("JPEG files", "*.jpg"),
                ("All files", "*.*")
            ]
        )
        if file_path:
            decrypted_img_rgb.save(file_path)
            messagebox.showinfo("Saved", f"Decrypted image saved to:\n{file_path}")
            log_message(f"Decrypted image saved: {os.path.basename(file_path)}")

def clear_all():
    """Clear all data and reset UI"""
    global selected_image_path, canvas_widget
    selected_image_path = None
    
    if canvas_widget:
        canvas_widget.get_tk_widget().destroy()
        canvas_widget = None
    
    # Reset UI elements
    label_path.config(text="No image selected")
    preview_label.config(text="No preview", image='')
    info_original.config(text="No image loaded")
    key_status.config(text="Enter 8 characters")
    
    # Clear results frame
    for widget in results_frame.winfo_children():
        widget.destroy()
    
    # Clear log
    if 'log_text' in globals():
        log_text.delete(1.0, tk.END)
    
    # Disable buttons
    btn_save_encrypted.config(state="disabled")
    btn_save_decrypted.config(state="disabled")
    process_button.config(state="disabled")
    
    status_label.config(text="Ready")
    log_message("All data cleared")

# ================= GUI SETUP =================
root = tk.Tk()
root.title("DES Image Encryption Analyzer")
root.geometry("1400x1000")

# Configure style
style = ttk.Style()
style.theme_use('clam')

# Configure colors
bg_color = "#f5f5f5"
header_color = "#2c3e50"
accent_color = "#3498db"

# Set window background
root.configure(bg=bg_color)

# Header
header_frame = tk.Frame(root, bg=header_color, height=120)
header_frame.pack(fill="x")
header_frame.pack_propagate(False)

tk.Label(header_frame, text="🔐 Enkripsi Gambar dengan Algoritma DES", 
         font=("Arial", 22, "bold"), fg="white", bg=header_color).pack(pady=(15, 5))
tk.Label(header_frame, text="Enkripsi dan  Dekripsi ", 
         font=("Arial", 11), fg="#ecf0f1", bg=header_color).pack()

# Main container
main_container = tk.Frame(root, bg=bg_color)
main_container.pack(fill="both", expand=True, padx=20, pady=15)

# Left panel - Controls
left_panel = tk.Frame(main_container, bg=bg_color, width=350)
left_panel.pack(side="left", fill="y", padx=(0, 15))

# Control frame
control_frame = tk.LabelFrame(left_panel, text="🛠️ Controls", font=("Arial", 11, "bold"),
                            bg=bg_color, padx=15, pady=15, relief="groove")
control_frame.pack(fill="x", pady=(0, 15))

# Image upload section
upload_frame = tk.Frame(control_frame, bg=bg_color)
upload_frame.pack(fill="x", pady=(0, 15))

tk.Label(upload_frame, text="Image File:", font=("Arial", 10, "bold"),
        bg=bg_color).pack(anchor="w")
btn_upload = tk.Button(upload_frame, text="📁 Upload Gambar", 
                      command=upload_image, width=25, height=1,
                      font=("Arial", 10), bg="#27ae60", fg="white",
                      activebackground="#229954")
btn_upload.pack(pady=5)

label_path = tk.Label(upload_frame, text="No image selected", 
                     font=("Arial", 9), fg="#7f8c8d", bg=bg_color, wraplength=300)
label_path.pack()

# Preview section
preview_frame = tk.Frame(control_frame, bg=bg_color)
preview_frame.pack(fill="x", pady=(0, 15))

tk.Label(preview_frame, text="Preview:", font=("Arial", 10, "bold"),
        bg=bg_color).pack(anchor="w")

preview_label = tk.Label(preview_frame, text="No preview", 
                        font=("Arial", 9), fg="gray", bg="white",
                        width=25, height=10, relief="sunken")
preview_label.pack(pady=5)

info_original = tk.Label(preview_frame, text="No image loaded", 
                        font=("Arial", 8), fg="#7f8c8d", bg=bg_color)
info_original.pack()

# Key input section
key_frame = tk.Frame(control_frame, bg=bg_color)
key_frame.pack(fill="x", pady=(0, 15))

tk.Label(key_frame, text="Encryption Key:", font=("Arial", 10, "bold"),
        bg=bg_color).pack(anchor="w")

key_input_frame = tk.Frame(key_frame, bg=bg_color)
key_input_frame.pack(fill="x")

entry_key = tk.Entry(key_input_frame, font=("Courier", 11), width=20)
entry_key.insert(0, "12345678")
entry_key.pack(side="left", padx=(0, 5))
entry_key.bind("<KeyRelease>", lambda e: validate_key())

btn_toggle_key = tk.Button(key_input_frame, text="👁 Show", 
                          command=toggle_key_visibility, width=8)
btn_toggle_key.pack(side="left")

key_status = tk.Label(key_frame, text="Masukkan 8 karakter", 
                     font=("Arial", 9), bg=bg_color, fg="green")
key_status.pack(anchor="w", pady=(5, 0))

# Action buttons
action_frame = tk.Frame(control_frame, bg=bg_color)
action_frame.pack(fill="x", pady=10)

process_button = tk.Button(action_frame, text="🔐 Encrypt & Analyze", 
                          command=process_image, width=20, height=1,
                          font=("Arial", 10, "bold"), bg="#3498db", fg="white",
                          activebackground="#2980b9", state="disabled")
process_button.pack(pady=5)

btn_clear = tk.Button(action_frame, text="🗑️ Clear All", 
                     command=clear_all, width=20, height=1,
                     font=("Arial", 10), bg="#95a5a6", fg="white",
                     activebackground="#7f8c8d")
btn_clear.pack(pady=5)

# Progress bar
progress_frame = tk.Frame(control_frame, bg=bg_color)
progress_frame.pack(fill="x", pady=10)

progress_bar = ttk.Progressbar(progress_frame, mode='indeterminate', length=280)
progress_bar.pack()

status_label = tk.Label(progress_frame, text="Ready", font=("Arial", 9),
                       bg=bg_color, fg="#2c3e50")
status_label.pack(pady=(5, 0))

# Save buttons
save_frame = tk.Frame(control_frame, bg=bg_color)
save_frame.pack(fill="x", pady=10)

btn_save_encrypted = tk.Button(save_frame, text="💾 Save Encrypted", 
                              command=save_encrypted_image, width=14,
                              state="disabled", bg="#e67e22", fg="white")
btn_save_encrypted.pack(side="left", padx=2)

btn_save_decrypted = tk.Button(save_frame, text="💾 Save Decrypted", 
                              command=save_decrypted_image, width=14,
                              state="disabled", bg="#9b59b6", fg="white")
btn_save_decrypted.pack(side="left", padx=2)

# Results frame
results_frame = tk.LabelFrame(left_panel, text="📊 Metrics Results", 
                            font=("Arial", 11, "bold"), bg=bg_color,
                            padx=10, pady=10, relief="groove")
results_frame.pack(fill="both", expand=True)

# Right panel - Visualization
right_panel = tk.Frame(main_container, bg=bg_color)
right_panel.pack(side="right", fill="both", expand=True)

# Visualization frame
viz_frame = tk.LabelFrame(right_panel, text="📈 Visualization", 
                         font=("Arial", 11, "bold"), bg=bg_color,
                         padx=10, pady=10, relief="groove")
viz_frame.pack(fill="both", expand=True)

# Log frame at bottom
log_frame = tk.Frame(root, bg="#2c3e50", height=150)
log_frame.pack(fill="x", side="bottom")
log_frame.pack_propagate(False)

tk.Label(log_frame, text="📝 Analysis Log", font=("Arial", 10, "bold"),
        fg="white", bg="#2c3e50").pack(anchor="w", padx=10, pady=(5, 0))

# Create scrolled text widget for log
log_text = scrolledtext.ScrolledText(log_frame, height=6, font=("Courier", 9),
                                    bg="#1a252f", fg="#ecf0f1", insertbackground="white")
log_text.pack(fill="both", expand=True, padx=10, pady=(0, 10))

# Add initial log message
log_message("Application started. Ready to analyze images.")
log_message("Instructions: Upload image → Enter 8-character key → Click 'Encrypt & Analyze'")

# Center window on screen
root.update_idletasks()
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
window_width = root.winfo_width()
window_height = root.winfo_height()
x = (screen_width - window_width) // 2
y = (screen_height - window_height) // 2
root.geometry(f"+{x}+{y}")

# Start the application
root.mainloop()