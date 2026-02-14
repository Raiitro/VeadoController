import os
import sys
import subprocess

# 1. Détection dynamique de l'installation Python source
python_base = sys.base_prefix
dlls_dir = os.path.join(python_base, 'DLLs')
tcl_dir = os.path.join(python_base, 'tcl')

# 2. Construction de la commande PyInstaller
# On force l'inclusion des fichiers que PyInstaller "rate" sur Python 3.13
cmd = [
    'pyinstaller',
    '--noconsole',
    '--onefile',
    '--name=VeadoController',
    '--collect-all=mediapipe',
    # Inclusion des modèles IA
    f'--add-data=models{os.pathsep}models',
    # Greffe forcée de Tkinter (Lib & DLLs)
    f'--add-data={os.path.join(tcl_dir, "tcl8.6")}{os.pathsep}tcl_data',
    f'--add-data={os.path.join(tcl_dir, "tk8.6")}{os.pathsep}tk_data',
    f'--add-binary={os.path.join(dlls_dir, "_tkinter.pyd")}{os.pathsep}.',
    f'--add-binary={os.path.join(dlls_dir, "tcl86t.dll")}{os.pathsep}.',
    f'--add-binary={os.path.join(dlls_dir, "tk86t.dll")}{os.pathsep}.',
    # Script source
    os.path.join('src', 'main.py')
]

print(f"--- Démarrage du build avec Python {sys.version} ---")
subprocess.run(cmd)
print("--- Build terminé ! Vérifie le dossier /dist ---")