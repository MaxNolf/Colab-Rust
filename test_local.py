import os
import sys

# Adaptação para poder importar colab_rust_bridge do diretório atual
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from colab_rust_bridge import pull

def test_bridge():
    print("=== Iniciando teste local do Colab-Rust-Bridge ===")
    
    # O diretório do template_github local
    current_dir = os.path.abspath(os.path.dirname(__file__))
    template_path = os.path.join(current_dir, "template_github")
    
    try:
        # A bridge vai copiar/clonar o diretorio e compilar
        modulo_rust = pull(template_path, "template_rust_module")
        
        resultado = modulo_rust.soma_rapida(10, 32)
        print(f"Resultado da soma_rapida(10, 32) via Rust é: {resultado}")
        
        if resultado == 42:
            print("=== Teste finalizado com SUCESSO! ===")
        else:
            print("=== Teste FALHOU! Valor inesperado. ===")
            
    except Exception as e:
        print(f"=== Teste FALHOU com exceção: {e} ===")

if __name__ == "__main__":
    test_bridge()
