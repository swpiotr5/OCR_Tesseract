#!/usr/bin/env python3
"""
Skrypt budowania EXE z lokalnym Tesseract-OCR
Automatycznie kopiuje Tesseract-OCR obok EXE w dist/
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path

def check_requirements():
    """Sprawdź wymagania do budowania"""
    print("🔍 Sprawdzanie wymagań...")
    
    # Sprawdź PyInstaller
    try:
        import PyInstaller
        print(f"✅ PyInstaller: {PyInstaller.__version__}")
    except ImportError:
        print("❌ PyInstaller nie jest zainstalowany!")
        print("Uruchom: pip install pyinstaller")
        return False
    
    script_dir = Path(__file__).parent
    tesseract_dir = script_dir / "Tesseract-OCR"
    tesseract_exe = tesseract_dir / "tesseract.exe"
    
    if not tesseract_exe.exists():
        print(f"❌ Lokalny Tesseract nie znaleziony: {tesseract_exe}")
        print("Upewnij się, że folder Tesseract-OCR jest obok main.py")
        return False
    
    print(f"✅ Lokalny Tesseract: {tesseract_exe}")
    
    tessdata_dir = tesseract_dir / "tessdata"
    if not tessdata_dir.exists():
        print(f"❌ Folder tessdata nie znaleziony: {tessdata_dir}")
        return False
    
    eng_file = tessdata_dir / "eng.traineddata"
    pol_file = tessdata_dir / "pol.traineddata"
    
    if eng_file.exists():
        print(f"✅ Model angielski: {eng_file}")
    else:
        print(f"⚠️ Brak modelu angielskiego: {eng_file}")
    
    if pol_file.exists():
        print(f"✅ Model polski: {pol_file}")
    else:
        print(f"⚠️ Brak modelu polskiego: {pol_file}")
    
    return True

def clean_build():
    print("\n🧹 Czyszczenie poprzednich builds...")
    
    dirs_to_clean = ['build', 'dist', '__pycache__']
    for dir_name in dirs_to_clean:
        if os.path.exists(dir_name):
            shutil.rmtree(dir_name)
            print(f"✅ Usunięto: {dir_name}")

def build_exe():
    print("\n🔨 Budowanie EXE...")
    
    pyinstaller_cmd = [
        'pyinstaller',
        '--onefile',           
        '--noconsole',          
        '--name=OCR_Tesseract_Pro',  
        '--icon=NONE',        

        '--hidden-import=PIL._tkinter_finder',
        '--hidden-import=tkinter',
        '--hidden-import=tkinter.ttk',
        '--hidden-import=tkinter.filedialog',
        '--hidden-import=tkinter.messagebox',
        '--hidden-import=tkinter.scrolledtext',
        '--hidden-import=pytesseract',
        '--hidden-import=cv2',
        '--hidden-import=numpy',
        '--hidden-import=pandas',

        '--collect-all=tkinter',
        '--copy-metadata=pillow',
        '--copy-metadata=opencv-python',
        
        'main.py'
    ]
    
    try:
        result = subprocess.run(pyinstaller_cmd, check=True, capture_output=True, text=True)
        print("✅ PyInstaller zakończony pomyślnie")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Błąd PyInstaller: {e}")
        print(f"Stdout: {e.stdout}")
        print(f"Stderr: {e.stderr}")
        return False

def copy_tesseract():
    print("\n📁 Kopiowanie Tesseract-OCR do dist/...")
    
    script_dir = Path(__file__).parent
    source_tesseract = script_dir / "Tesseract-OCR"
    dest_tesseract = script_dir / "dist" / "Tesseract-OCR"
    
    if not source_tesseract.exists():
        print(f"❌ Źródłowy folder Tesseract nie istnieje: {source_tesseract}")
        return False
    
    try:
        if dest_tesseract.exists():
            shutil.rmtree(dest_tesseract)

        shutil.copytree(source_tesseract, dest_tesseract)
        print(f"✅ Skopiowano Tesseract-OCR do: {dest_tesseract}")

        tesseract_exe = dest_tesseract / "tesseract.exe"
        if tesseract_exe.exists():
            print(f"✅ tesseract.exe: {tesseract_exe}")
        else:
            print(f"❌ Brak tesseract.exe w: {tesseract_exe}")
            return False

        tessdata_dir = dest_tesseract / "tessdata"
        if tessdata_dir.exists():
            lang_files = list(tessdata_dir.glob("*.traineddata"))
            print(f"✅ tessdata: {len(lang_files)} plików językowych")
            for lang_file in lang_files:
                print(f"   - {lang_file.name}")
        else:
            print(f"❌ Brak folderu tessdata w: {tessdata_dir}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Błąd podczas kopiowania: {e}")
        return False

def create_test_script():
    print("\n📝 Tworzenie skryptu testowego...")
    
    test_script = '''@echo off
echo =================================
echo    Test OCR Tesseract Pro
echo =================================
echo.

echo Sprawdzanie struktury plików...
if exist "OCR_Tesseract_Pro.exe" (
    echo ✅ OCR_Tesseract_Pro.exe
) else (
    echo ❌ Brak OCR_Tesseract_Pro.exe
    goto :error
)

if exist "Tesseract-OCR\\tesseract.exe" (
    echo ✅ Tesseract-OCR\\tesseract.exe
) else (
    echo ❌ Brak Tesseract-OCR\\tesseract.exe
    goto :error
)

if exist "Tesseract-OCR\\tessdata" (
    echo ✅ Tesseract-OCR\\tessdata
) else (
    echo ❌ Brak Tesseract-OCR\\tessdata
    goto :error
)

echo.
echo Uruchamianie aplikacji...
echo Jeśli aplikacja się uruchomi, test przebiegł pomyślnie!
echo.
start OCR_Tesseract_Pro.exe

echo.
echo Test zakończony!
pause
goto :end

:error
echo.
echo ❌ Test nie powiódł się!
echo Sprawdź czy wszystkie pliki zostały skopiowane poprawnie.
pause

:end
'''
    
    test_file = Path(__file__).parent / "dist" / "test_app.bat"
    try:
        with open(test_file, 'w', encoding='cp1250') as f:
            f.write(test_script)
        print(f"✅ Utworzono: {test_file}")
        return True
    except Exception as e:
        print(f"❌ Błąd tworzenia skryptu testowego: {e}")
        return False

def main():
    print("🚀 Budowanie OCR Tesseract Pro z lokalnym Tesseract-OCR")
    print("=" * 60)
    
    if not check_requirements():
        print("\n❌ Nie można kontynuować - brak wymagań")
        input("Naciśnij Enter aby zakończyć...")
        return

    clean_build()

    if not build_exe():
        print("\n❌ Budowanie EXE nie powiodło się")
        input("Naciśnij Enter aby zakończyć...")
        return

    if not copy_tesseract():
        print("\n❌ Kopiowanie Tesseract nie powiodło się")
        input("Naciśnij Enter aby zakończyć...")
        return

    create_test_script()
    
    print("\n" + "=" * 60)
    print("🎉 BUDOWANIE ZAKOŃCZONE POMYŚLNIE!")
    print("=" * 60)
    print(f"📁 Lokalizacja: {Path(__file__).parent / 'dist'}")
    print("📱 Plik EXE: OCR_Tesseract_Pro.exe")
    print("🔧 Tesseract: Tesseract-OCR/")
    print("🧪 Test: test_app.bat")
    print("\n💡 Aby przetestować aplikację:")
    print("   1. Przejdź do folderu dist/")
    print("   2. Uruchom test_app.bat")
    print("   3. Lub uruchom bezpośrednio OCR_Tesseract_Pro.exe")
    
    input("\nNaciśnij Enter aby zakończyć...")

if __name__ == "__main__":
    main()