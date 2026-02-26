import os
import sys
import shutil
import subprocess
import importlib
import tempfile

def pull(github_url: str, module_name: str):
    print("Iniciando Colab-Rust-Bridge...")
    
    # Resolvendo o diretório temporário
    # Utilizando /tmp/rust_bridge conforme solicitado
    if sys.platform == "win32":
        base_tmp = tempfile.gettempdir()
        tmp_dir = os.path.join(base_tmp, "rust_bridge")
    else:
        tmp_dir = "/tmp/rust_bridge"
    
    os.makedirs(tmp_dir, exist_ok=True)
    
    project_dir = os.path.join(tmp_dir, module_name)
    if os.path.exists(project_dir):
        # Permite resolver locks no Windows caso existam
        shutil.rmtree(project_dir, ignore_errors=True)
        
    print("Clonando...")
    try:
        if os.path.isdir(github_url):
            # Para o script test_local.py que usa um caminho local
            shutil.copytree(github_url, project_dir)
        else:
            subprocess.run(
                ["git", "clone", github_url, project_dir],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True
            )
    except subprocess.CalledProcessError:
        raise RuntimeError(f"Erro ao clonar o repositório: {github_url}. Verifique se a URL é válida.")
    except Exception as e:
        raise RuntimeError(f"Erro ao obter o código fonte: {e}")

    # Verifica se maturin está disponível via subprocesso
    maturin_installed = shutil.which("maturin") is not None
    if not maturin_installed:
        try:
            # Verifica fallback invocando como modulo do python
            subprocess.run([sys.executable, "-m", "maturin", "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            maturin_installed = True
        except:
            maturin_installed = False

    if not maturin_installed:
        print("Instalando maturin (isso ocorre apenas na primeira vez)...")
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "maturin"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True
            )
        except subprocess.CalledProcessError:
            raise RuntimeError("Erro ao instalar o pacote maturin via pip.")

    print("Compilando (isso pode levar 1 minuto)...")
    try:
        # Usando sys.executable -m maturin para garantir que usamos a versão instalada no python atual
        subprocess.run(
            [sys.executable, "-m", "maturin", "build", "--release"],
            cwd=project_dir,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True
        )
    except subprocess.CalledProcessError:
        try:
            # Tenta rodar maturin diretamente caso o anterior falhe por alguma peculiaridade no path
            subprocess.run(
                ["maturin", "build", "--release"],
                cwd=project_dir,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True
            )
        except subprocess.CalledProcessError:
            raise RuntimeError("Erro de compilação no Rust. Verifique os logs e o código-fonte.")

    wheels_dir = os.path.join(project_dir, "target", "wheels")
    if not os.path.exists(wheels_dir):
        raise RuntimeError(f"O diretório de wheels não foi encontrado em {wheels_dir}")
        
    wheels = [f for f in os.listdir(wheels_dir) if f.endswith(".whl")]
    if not wheels:
        raise RuntimeError("Nenhum arquivo .whl foi gerado pelo maturin.")
        
    wheel_path = os.path.join(wheels_dir, wheels[0])

    print("Injetando módulo no Python...")
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", wheel_path, "--force-reinstall"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True
        )
    except subprocess.CalledProcessError:
        raise RuntimeError("Erro ao instalar o módulo compilado via pip.")

    print("Injetado com sucesso!")
    
    # Invalidate cache e carrega o módulo
    importlib.invalidate_caches()
    if module_name in sys.modules:
        return importlib.reload(sys.modules[module_name])
    else:
        return importlib.import_module(module_name)
