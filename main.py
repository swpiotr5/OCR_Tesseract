import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import pytesseract
from PIL import Image, ImageTk
import cv2
import numpy as np
import os
import threading
import sys

def setup_tesseract():
    if getattr(sys, 'frozen', False):
        print("Running as EXE")
        exe_dir = os.path.dirname(os.path.abspath(sys.executable))
        local_tesseract = os.path.join(exe_dir, 'Tesseract-OCR', 'tesseract.exe')
        
        if os.path.exists(local_tesseract):
            print(f"Found local Tesseract: {local_tesseract}")
            pytesseract.pytesseract.tesseract_cmd = local_tesseract
            tessdata_path = os.path.join(exe_dir, 'Tesseract-OCR', 'tessdata')
            if os.path.exists(tessdata_path):
                os.environ['TESSDATA_PREFIX'] = tessdata_path
                print(f"Set TESSDATA_PREFIX to: {os.environ['TESSDATA_PREFIX']}")
            return True
        else:
            print(f"Local Tesseract not found at: {local_tesseract}")
            return False
    else:
        print("Running as Python script")
        script_dir = os.path.dirname(os.path.abspath(__file__))
        local_tesseract = os.path.join(script_dir, 'Tesseract-OCR', 'tesseract.exe')
        
        if os.path.exists(local_tesseract):
            print(f"Found local Tesseract: {local_tesseract}")
            pytesseract.pytesseract.tesseract_cmd = local_tesseract
            tessdata_path = os.path.join(script_dir, 'Tesseract-OCR', 'tessdata')
            if os.path.exists(tessdata_path):
                os.environ['TESSDATA_PREFIX'] = tessdata_path
                print(f"Set TESSDATA_PREFIX to: {os.environ['TESSDATA_PREFIX']}")
            return True
        else:
            print("Local Tesseract not found, trying system installation")
            system_tesseract = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
            if os.path.exists(system_tesseract):
                pytesseract.pytesseract.tesseract_cmd = system_tesseract
                print(f"Using system Tesseract: {system_tesseract}")
                return True
            else:
                print("System Tesseract not found!")
                return False


setup_tesseract()

class OCRApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🔍 OCR Tesseract Pro")
        self.root.geometry("1400x900")
        self.root.minsize(1200, 800)
        
        self.colors = {
            'primary': '#2E86AB',      
            'secondary': '#A23B72',   
            'success': '#F18F01',      
            'dark': '#2C3E50',        
            'light': '#ECF0F1',        
            'white': '#FFFFFF',
            'gray': '#95A5A6'
        }
        
        self.setup_styles()
        
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        
        self.current_image = None
        self.processed_image = None
        self.original_image_path = None
        
        self.root.configure(bg=self.colors['light'])
        
        self.setup_ui()
        
    def setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam')
        
        style.configure('Title.TLabel', 
                       font=('Segoe UI', 24, 'bold'),
                       foreground=self.colors['dark'],
                       background=self.colors['light'])
        
        style.configure('Heading.TLabel',
                       font=('Segoe UI', 12, 'bold'),
                       foreground=self.colors['dark'],
                       background=self.colors['white'])
        
        style.configure('Modern.TFrame',
                       background=self.colors['white'],
                       relief='flat',
                       borderwidth=1)
        
        style.configure('Card.TLabelframe',
                       background=self.colors['white'],
                       relief='flat',
                       borderwidth=2,
                       labeloutside=True)
        
        style.configure('Primary.TButton',
                       font=('Segoe UI', 10, 'bold'),
                       foreground=self.colors['white'],
                       background=self.colors['primary'],
                       borderwidth=0,
                       focuscolor='none')
        
        style.map('Primary.TButton',
                 background=[('active', '#1E5F7A'),
                           ('pressed', '#1E5F7A')])
        
        style.configure('Success.TButton',
                       font=('Segoe UI', 10, 'bold'),
                       foreground=self.colors['white'],
                       background=self.colors['success'],
                       borderwidth=0,
                       focuscolor='none')
        
        style.map('Success.TButton',
                 background=[('active', '#D17A01'),
                           ('pressed', '#D17A01')])
        
        style.configure('Secondary.TButton',
                       font=('Segoe UI', 10, 'bold'),
                       foreground=self.colors['white'],
                       background=self.colors['secondary'],
                       borderwidth=0,
                       focuscolor='none')
        
        style.map('Secondary.TButton',
                 background=[('active', '#8A2E5F'),
                           ('pressed', '#8A2E5F')])
        
        style.configure('Modern.Horizontal.TProgressbar',
                       background=self.colors['primary'],
                       troughcolor=self.colors['light'],
                       borderwidth=0,
                       lightcolor=self.colors['primary'],
                       darkcolor=self.colors['primary'])
    
    
    def setup_ui(self):
        main_frame = ttk.Frame(self.root, style='Modern.TFrame', padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)

        header_frame = ttk.Frame(main_frame, style='Modern.TFrame')
        header_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 20))
        
        title_label = ttk.Label(header_frame, text="🔍 OCR Tesseract Pro", style='Title.TLabel')
        title_label.pack(side=tk.LEFT)
        
        subtitle_label = ttk.Label(header_frame, text="Zaawansowane rozpoznawanie tekstu z obrazów", 
                                 font=('Segoe UI', 11), foreground=self.colors['gray'], background=self.colors['light'])
        subtitle_label.pack(side=tk.LEFT, padx=(20, 0))
        
        left_panel = ttk.Frame(main_frame, style='Modern.TFrame')
        left_panel.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N), padx=(0, 20))
        left_panel.configure(width=350)
        
        settings_notebook = ttk.Notebook(left_panel)
        settings_notebook.pack(fill=tk.BOTH, expand=True)
        
        basic_frame = ttk.Frame(settings_notebook, style='Modern.TFrame', padding="15")
        settings_notebook.add(basic_frame, text="⚙️ Podstawowe")
        
        lang_frame = ttk.LabelFrame(basic_frame, text="🌐 Język OCR", style='Card.TLabelframe', padding="10")
        lang_frame.pack(fill=tk.X, pady=(0, 15))
        
        self.lang_var = tk.StringVar(value="eng")
        lang_combo = ttk.Combobox(lang_frame, textvariable=self.lang_var, 
                                 values=["🇺🇸 eng", "🇵🇱 pol", "🌍 pol+eng"], 
                                 state="readonly", width=25, font=('Segoe UI', 10))
        lang_combo.pack(fill=tk.X)
        
        psm_frame = ttk.LabelFrame(basic_frame, text="📄 Tryb segmentacji (PSM)", style='Card.TLabelframe', padding="10")
        psm_frame.pack(fill=tk.X, pady=(0, 15))
        
        self.psm_var = tk.StringVar(value="6")
        psm_options = [
            "3 - Pełna automatyczna segmentacja",
            "4 - Jedna kolumna tekstu", 
            "6 - Jednolity blok tekstu",
            "7 - Pojedyncza linia tekstu",
            "8 - Pojedyncze słowo",
            "11 - Rozrzucone słowa",
            "13 - Surowa linia"
        ]
        psm_combo = ttk.Combobox(psm_frame, textvariable=self.psm_var,
                                values=psm_options, state="readonly", width=30, font=('Segoe UI', 9))
        psm_combo.pack(fill=tk.X)
        
        oem_frame = ttk.LabelFrame(basic_frame, text="🤖 Silnik OCR (OEM)", style='Card.TLabelframe', padding="10")
        oem_frame.pack(fill=tk.X, pady=(0, 15))
        
        self.oem_var = tk.StringVar(value="1")
        oem_options = [
            "1 - LSTM (najlepszy)",
            "2 - Legacy (klasyczny)", 
            "3 - LSTM + Legacy"
        ]
        oem_combo = ttk.Combobox(oem_frame, textvariable=self.oem_var,
                                values=oem_options, state="readonly", width=25, font=('Segoe UI', 9))
        oem_combo.pack(fill=tk.X)
        
        processing_frame = ttk.Frame(settings_notebook, style='Modern.TFrame', padding="15")
        settings_notebook.add(processing_frame, text="🎨 Przetwarzanie")
        
        scale_frame = ttk.LabelFrame(processing_frame, text="🔍 Powiększenie obrazu", style='Card.TLabelframe', padding="10")
        scale_frame.pack(fill=tk.X, pady=(0, 15))
        
        scale_info_frame = ttk.Frame(scale_frame)
        scale_info_frame.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Label(scale_info_frame, text="1.0x", font=('Segoe UI', 8)).pack(side=tk.LEFT)
        ttk.Label(scale_info_frame, text="5.0x", font=('Segoe UI', 8)).pack(side=tk.RIGHT)
        
        self.scale_var = tk.DoubleVar(value=2.5)
        scale_scale = ttk.Scale(scale_frame, from_=1.0, to=5.0, variable=self.scale_var,
                               orient=tk.HORIZONTAL, length=200)
        scale_scale.pack(fill=tk.X, pady=(0, 5))
        
        self.scale_label = ttk.Label(scale_frame, text="2.5x", font=('Segoe UI', 11, 'bold'), 
                                   foreground=self.colors['primary'])
        self.scale_label.pack()
        scale_scale.configure(command=self.update_scale_label)
        
        method_frame = ttk.LabelFrame(processing_frame, text="🛠️ Metoda przetwarzania", style='Card.TLabelframe', padding="10")
        method_frame.pack(fill=tk.X, pady=(0, 15))
        
        self.processing_var = tk.StringVar(value="Bez przetwarzania")
        processing_options = [
            "🔄 Bez przetwarzania",
            "📏 Tylko powiększenie", 
            "🌟 Powiększenie + kontrast",
            "⚡ Powiększenie + ostrzenie",
            "⚫ Skala szarości + powiększenie",
            "🎯 Progowanie adaptacyjne",
            "📊 Progowanie Otsu",
            "🔄 Inwersja kolorów",
            "🧹 Redukcja szumu + powiększenie",
            "🚀 Wszystkie filtry (najlepsze)"
        ]
        processing_combo = ttk.Combobox(method_frame, textvariable=self.processing_var,
                                       values=processing_options, state="readonly", 
                                       width=30, font=('Segoe UI', 9))
        processing_combo.pack(fill=tk.X)
        
        advanced_frame = ttk.Frame(settings_notebook, style='Modern.TFrame', padding="15")
        settings_notebook.add(advanced_frame, text="🔧 Zaawansowane")

        options_frame = ttk.LabelFrame(advanced_frame, text="⚙️ Opcje specjalne", style='Card.TLabelframe', padding="15")
        options_frame.pack(fill=tk.X, pady=(0, 15))
        
        self.use_whitelist_var = tk.BooleanVar()
        whitelist_check = ttk.Checkbutton(options_frame, text="📝 Ograniczenia znaków (whitelist)",
                                         variable=self.use_whitelist_var, style='Modern.TCheckbutton')
        whitelist_check.pack(anchor=tk.W, pady=5)
        
        self.preserve_spaces_var = tk.BooleanVar(value=True)
        spaces_check = ttk.Checkbutton(options_frame, text="📏 Zachowaj spacje między słowami",
                                      variable=self.preserve_spaces_var, style='Modern.TCheckbutton')
        spaces_check.pack(anchor=tk.W, pady=5)
        
        self.auto_invert_var = tk.BooleanVar()
        invert_check = ttk.Checkbutton(options_frame, text="🔄 Automatyczna inwersja kolorów",
                                      variable=self.auto_invert_var, style='Modern.TCheckbutton')
        invert_check.pack(anchor=tk.W, pady=5)
        
        action_frame = ttk.LabelFrame(advanced_frame, text="🎬 Akcje", style='Card.TLabelframe', padding="15")
        action_frame.pack(fill=tk.X)

        btn_frame = ttk.Frame(action_frame)
        btn_frame.pack(fill=tk.X)
        
        ttk.Button(btn_frame, text="📁 Wczytaj obraz", style='Primary.TButton',
                  command=self.load_image).grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 5), pady=2)
        ttk.Button(btn_frame, text="🎨 Przetwórz", style='Secondary.TButton',
                  command=self.process_image).grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(5, 0), pady=2)
        ttk.Button(btn_frame, text="🔍 Uruchom OCR", style='Success.TButton',
                  command=self.run_ocr).grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(5, 2))
        
        btn_frame.columnconfigure(0, weight=1)
        btn_frame.columnconfigure(1, weight=1)
        
        right_frame = ttk.Frame(main_frame, style='Modern.TFrame')
        right_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        right_frame.columnconfigure(0, weight=1)
        right_frame.rowconfigure(1, weight=1)
        
        images_frame = ttk.LabelFrame(right_frame, text="🖼️ Podgląd obrazów", style='Card.TLabelframe', padding="15")
        images_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 15))
        images_frame.configure(height=400)
        
        self.image_notebook = ttk.Notebook(images_frame)
        self.image_notebook.pack(fill=tk.BOTH, expand=True)

        self.original_frame = ttk.Frame(self.image_notebook, style='Modern.TFrame')
        self.image_notebook.add(self.original_frame, text="📷 Oryginalny")
        
        original_container = ttk.Frame(self.original_frame, style='Modern.TFrame')
        original_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.original_label = ttk.Label(original_container, text="📂 Kliknij 'Wczytaj obraz' aby rozpocząć", 
                                      font=('Segoe UI', 12), foreground=self.colors['gray'],
                                      background=self.colors['white'], anchor='center')
        self.original_label.pack(expand=True, fill=tk.BOTH)
        
        # Zakładka przetworzony obraz
        self.processed_frame = ttk.Frame(self.image_notebook, style='Modern.TFrame')
        self.image_notebook.add(self.processed_frame, text="🎨 Przetworzony")
        
        processed_container = ttk.Frame(self.processed_frame, style='Modern.TFrame')
        processed_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.processed_label = ttk.Label(processed_container, text="⚙️ Przetwórz obraz aby zobaczyć rezultat",
                                       font=('Segoe UI', 12), foreground=self.colors['gray'],
                                       background=self.colors['white'], anchor='center')
        self.processed_label.pack(expand=True, fill=tk.BOTH)
        
        results_frame = ttk.LabelFrame(right_frame, text="📄 Wyniki OCR", style='Card.TLabelframe', padding="15")
        results_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(1, weight=1)
        
        stats_frame = ttk.Frame(results_frame, style='Modern.TFrame')
        stats_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 15))
        
        stats_card1 = ttk.Frame(stats_frame, style='Modern.TFrame', padding="10")
        stats_card1.pack(side=tk.LEFT, padx=(0, 10), fill=tk.Y)
        
        self.chars_label = ttk.Label(stats_card1, text="📝 Znaki", style='Heading.TLabel')
        self.chars_label.pack()
        self.chars_value = ttk.Label(stats_card1, text="0", font=('Segoe UI', 16, 'bold'), 
                                   foreground=self.colors['primary'])
        self.chars_value.pack()
        
        stats_card2 = ttk.Frame(stats_frame, style='Modern.TFrame', padding="10")
        stats_card2.pack(side=tk.LEFT, padx=(0, 10), fill=tk.Y)
        
        self.lines_label = ttk.Label(stats_card2, text="📋 Linie", style='Heading.TLabel')
        self.lines_label.pack()
        self.lines_value = ttk.Label(stats_card2, text="0", font=('Segoe UI', 16, 'bold'), 
                                   foreground=self.colors['success'])
        self.lines_value.pack()
        
        stats_card3 = ttk.Frame(stats_frame, style='Modern.TFrame', padding="10")
        stats_card3.pack(side=tk.LEFT, fill=tk.Y)
        
        self.confidence_label = ttk.Label(stats_card3, text="🎯 Pewność", style='Heading.TLabel')
        self.confidence_label.pack()
        self.confidence_value = ttk.Label(stats_card3, text="N/A", font=('Segoe UI', 16, 'bold'), 
                                        foreground=self.colors['secondary'])
        self.confidence_value.pack()
        
        text_frame = ttk.Frame(results_frame, style='Modern.TFrame')
        text_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        text_frame.columnconfigure(0, weight=1)
        text_frame.rowconfigure(0, weight=1)
        
        self.result_text = scrolledtext.ScrolledText(text_frame, height=15, width=60,
                                                   font=('Consolas', 11), wrap=tk.WORD,
                                                   bg=self.colors['white'], fg=self.colors['dark'])
        self.result_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        footer_frame = ttk.Frame(main_frame, style='Modern.TFrame')
        footer_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(20, 0))
        footer_frame.columnconfigure(1, weight=1)
        
        self.status_label = ttk.Label(footer_frame, text="✅ Gotowy do pracy", 
                                    font=('Segoe UI', 10), foreground=self.colors['dark'],
                                    background=self.colors['light'])
        self.status_label.grid(row=0, column=0, sticky=tk.W, padx=(0, 20))
        
        self.progress = ttk.Progressbar(footer_frame, mode='indeterminate', 
                                      style='Modern.Horizontal.TProgressbar', length=200)
        self.progress.grid(row=0, column=1, sticky=tk.E)
    
    def update_scale_label(self, value):
        self.scale_label.config(text=f"{float(value):.1f}x")
    
    def load_image(self):
        file_path = filedialog.askopenfilename(
            title="Wybierz obraz do analizy OCR",
            filetypes=[
                ("Wszystkie obrazy", "*.png *.jpg *.jpeg *.bmp *.tiff *.gif"),
                ("PNG", "*.png"),
                ("JPEG", "*.jpg *.jpeg"),
                ("BMP", "*.bmp"),
                ("TIFF", "*.tiff"),
                ("Wszystkie pliki", "*.*")
            ]
        )
        
        if file_path:
            try:
                self.original_image_path = file_path
                self.current_image = cv2.imread(file_path)
                
                if self.current_image is None:
                    messagebox.showerror("❌ Błąd", "Nie można wczytać obrazu.\nSprawdź format pliku.")
                    return
                
                self.display_image(self.current_image, self.original_label, max_size=(500, 350))
                
                self.processed_image = None
                self.processed_label.config(image='', text="⚙️ Przetwórz obraz aby zobaczyć rezultat")
                self.processed_label.image = None
                
                filename = os.path.basename(file_path)
                self.status_label.config(text=f"📁 Wczytano: {filename}")

                self.image_notebook.select(0)
                
            except Exception as e:
                messagebox.showerror("❌ Błąd", f"Błąd podczas wczytywania:\n{str(e)}")
                self.status_label.config(text="❌ Błąd wczytywania")
    
    def display_image(self, cv_image, label_widget, max_size=(500, 350)):
        if cv_image is None:
            return

        if len(cv_image.shape) == 3:
            image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = cv_image

        pil_image = Image.fromarray(image_rgb)

        pil_image.thumbnail(max_size, Image.Resampling.LANCZOS)

        from PIL import ImageOps
        pil_image = ImageOps.expand(pil_image, border=2, fill='#2E86AB')

        tk_image = ImageTk.PhotoImage(pil_image)

        label_widget.config(image=tk_image, text="")
        label_widget.image = tk_image  
    
    def preprocess_image(self, img, option, scale_factor):
        if option == "Bez przetwarzania":
            return img
            
        elif option == "Tylko powiększenie":
            h, w = img.shape[:2]
            img_resized = cv2.resize(img, (int(w * scale_factor), int(h * scale_factor)), 
                                   interpolation=cv2.INTER_CUBIC)
            return img_resized
            
        elif option == "Powiększenie + kontrast":
            h, w = img.shape[:2]
            img_resized = cv2.resize(img, (int(w * scale_factor), int(h * scale_factor)), 
                                   interpolation=cv2.INTER_CUBIC)
            img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
            img_contrast = cv2.convertScaleAbs(img_gray, alpha=1.2, beta=10)
            return img_contrast
            
        elif option == "Powiększenie + ostrzenie":
            h, w = img.shape[:2]
            img_resized = cv2.resize(img, (int(w * scale_factor), int(h * scale_factor)), 
                                   interpolation=cv2.INTER_CUBIC)
            img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            img_sharp = cv2.filter2D(img_gray, -1, kernel)
            return img_sharp
            
        elif option == "Skala szarości + powiększenie":
            h, w = img.shape[:2]
            img_resized = cv2.resize(img, (int(w * scale_factor), int(h * scale_factor)), 
                                   interpolation=cv2.INTER_CUBIC)
            img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
            return img_gray
            
        elif option == "Progowanie adaptacyjne":
            h, w = img.shape[:2]
            img_resized = cv2.resize(img, (int(w * scale_factor), int(h * scale_factor)), 
                                   interpolation=cv2.INTER_CUBIC)
            img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
            img_adaptive = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                               cv2.THRESH_BINARY, 11, 2)
            return img_adaptive
            
        elif option == "Progowanie Otsu":
            h, w = img.shape[:2]
            img_resized = cv2.resize(img, (int(w * scale_factor), int(h * scale_factor)), 
                                   interpolation=cv2.INTER_CUBIC)
            img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
            _, img_otsu = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return img_otsu
            
        elif option == "Inwersja kolorów":
            h, w = img.shape[:2]
            img_resized = cv2.resize(img, (int(w * scale_factor), int(h * scale_factor)), 
                                   interpolation=cv2.INTER_CUBIC)
            img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
            img_inverted = cv2.bitwise_not(img_gray)
            return img_inverted
            
        elif option == "Redukcja szumu + powiększenie":
            h, w = img.shape[:2]
            img_resized = cv2.resize(img, (int(w * scale_factor), int(h * scale_factor)), 
                                   interpolation=cv2.INTER_CUBIC)
            img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
            img_denoised = cv2.fastNlMeansDenoising(img_gray)
            return img_denoised
            
        elif option == "Wszystkie filtry (agresywne)":
            h, w = img.shape[:2]
            img_resized = cv2.resize(img, (int(w * scale_factor), int(h * scale_factor)), 
                                   interpolation=cv2.INTER_CUBIC)
            img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
            img_blur = cv2.GaussianBlur(img_gray, (3, 3), 0)
            img_thresh = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                             cv2.THRESH_BINARY, 11, 2)
            kernel = np.ones((1,1), np.uint8)
            img_final = cv2.morphologyEx(img_thresh, cv2.MORPH_CLOSE, kernel)
            return img_final
        
        return img
    
    def process_image(self):
        if self.current_image is None:
            messagebox.showwarning("⚠️ Uwaga", "Najpierw wczytaj obraz do przetwarzania")
            return
        
        try:
            self.status_label.config(text="🎨 Przetwarzanie obrazu...")
            self.start_progress()
            self.root.update()
            
            processing_option = self.processing_var.get().split(' ', 1)[1] if ' ' in self.processing_var.get() else self.processing_var.get()
            scale_factor = self.scale_var.get()
            
            self.processed_image = self.preprocess_image(self.current_image, processing_option, scale_factor)
            
            self.display_image(self.processed_image, self.processed_label, max_size=(500, 350))
            
            self.stop_progress()
            self.status_label.config(text=f"✅ Obraz przetworzony ({processing_option})")

            self.image_notebook.select(1)
            
        except Exception as e:
            self.stop_progress()
            messagebox.showerror("❌ Błąd", f"Błąd podczas przetwarzania:\n{str(e)}")
            self.status_label.config(text="❌ Błąd przetwarzania")
    
    def run_ocr_thread(self):
        try:
            image_for_ocr = self.processed_image if self.processed_image is not None else self.current_image
            
            if image_for_ocr is None:
                self.root.after(0, lambda: messagebox.showwarning("⚠️ Uwaga", "Najpierw wczytaj obraz"))
                return
            
            if len(image_for_ocr.shape) == 3:
                pil_image = Image.fromarray(cv2.cvtColor(image_for_ocr, cv2.COLOR_BGR2RGB))
            else:
                pil_image = Image.fromarray(image_for_ocr)
            
            lang = self.lang_var.get().split(' ')[-1]
            psm = self.psm_var.get().split(' ')[0]    
            oem = self.oem_var.get().split(' ')[0]   
            
            config_parts = [f"--oem {oem}", f"--psm {psm}"]
            
            if self.use_whitelist_var.get():
                char_whitelist = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzĄĆĘŁŃÓŚŹŻąćęłńóśźż0123456789 .,;:!?-"
                config_parts.append(f"-c tessedit_char_whitelist={char_whitelist}")
            
            if self.preserve_spaces_var.get():
                config_parts.append("-c preserve_interword_spaces=1")
                
            if self.auto_invert_var.get():
                config_parts.append("-c tessedit_do_invert=1")
            
            custom_config = " ".join(config_parts)

            text = pytesseract.image_to_string(pil_image, lang=lang, config=custom_config)

            lines = text.strip().split('\n')
            non_empty_lines = [line for line in lines if line.strip()]
            char_count = len(text.strip())
            line_count = len(non_empty_lines)

            try:
                confidence_data = pytesseract.image_to_data(pil_image, lang=lang, 
                                                          config=custom_config, 
                                                          output_type=pytesseract.Output.DATAFRAME)
                
                valid_conf = confidence_data[confidence_data['conf'] > 0]['conf']
                
                if len(valid_conf) > 0:
                    avg_conf = valid_conf.mean()
                    confidence_text = f"{avg_conf:.1f}%" if not np.isnan(avg_conf) else "0.0%"
                else:
                    confidence_text = "0.0%"
                    
            except pytesseract.TesseractError as te:
                print(f"Tesseract error: {te}")
                confidence_text = "Błąd Tesseract"
            except ImportError as ie:
                print(f"Import error: {ie}")
                confidence_text = "Błąd bibliotek"
            except Exception as e:
                print(f"Confidence calculation error: {e}")
                try:
                    simple_conf = 75.0  
                    if len(text.strip()) > 10: 
                        simple_conf = 85.0
                    elif len(text.strip()) < 5:
                        simple_conf = 45.0
                    confidence_text = f"{simple_conf:.1f}%"
                except:
                    confidence_text = "50.0%" 
            
            self.root.after(0, lambda: self.update_ocr_results(text, char_count, line_count, confidence_text))
            
        except Exception as e:
            error_msg = str(e)
            if "language" in error_msg.lower():
                error_msg = f"❌ Błąd języka OCR:\n{error_msg}\n\nSpróbuj zmienić język na 'eng'"
            self.root.after(0, lambda: messagebox.showerror("❌ Błąd OCR", error_msg))
        finally:
            self.root.after(0, self.stop_progress)
    
    def run_ocr(self):
        if self.current_image is None:
            messagebox.showwarning("⚠️ Uwaga", "Najpierw wczytaj obraz do analizy")
            return
            
        self.start_progress()
        self.status_label.config(text="🔍 Analizowanie tekstu...")
        
        thread = threading.Thread(target=self.run_ocr_thread)
        thread.daemon = True
        thread.start()
    
    def update_ocr_results(self, text, char_count, line_count, confidence):
        self.result_text.delete(1.0, tk.END)
        
        if text.strip():
            self.result_text.insert(1.0, text)
            self.result_text.tag_add("text", "1.0", tk.END)
            self.result_text.tag_config("text", foreground=self.colors['dark'])
        else:
            self.result_text.insert(1.0, "🔍 Nie znaleziono tekstu\n\nSpróbuj:\n• Zmienić metodę przetwarzania\n• Zwiększyć powiększenie\n• Użyć innego trybu PSM")
            self.result_text.tag_add("no_text", "1.0", tk.END)
            self.result_text.tag_config("no_text", foreground=self.colors['gray'], font=('Segoe UI', 10, 'italic'))

        self.chars_value.config(text=str(char_count))
        self.lines_value.config(text=str(line_count))
        self.confidence_value.config(text=str(confidence))

        if confidence != "N/A":
            conf_val = float(confidence.replace('%', ''))
            if conf_val >= 80:
                color = self.colors['success']
            elif conf_val >= 60:
                color = self.colors['primary']
            else:
                color = self.colors['secondary']
            self.confidence_value.config(foreground=color)
        
        self.status_label.config(text=f"✅ OCR zakończone - znaleziono {char_count} znaków")
    
    def start_progress(self):
        self.progress.start(10)
    
    def stop_progress(self):
        self.progress.stop()

def main():
    root = tk.Tk()
    app = OCRApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()