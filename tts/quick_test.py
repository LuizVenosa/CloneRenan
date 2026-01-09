#!/usr/bin/env python3
"""
Teste rápido para validar correções
"""

def test_filters():
    """Testa filtros de texto"""
    print("\n" + "="*60)
    print("TESTE 1: Filtros de Texto")
    print("="*60 + "\n")
    
    from tts_filters import TTSTextFilter
    
    filter = TTSTextFilter()
    
    # Casos de teste
    test_cases = [
        ("Veja https://youtube.com/watch?v=123 este vídeo.", "URLs"),
        ("**negrito** e *itálico* com `código`", "Markdown"),
        ("Fonte 1 (Vídeo sobre política):", "Fonte"),
        ("Link: https://youtu.be/abc", "Link"),
        ("Análise ｜ 31⧸07⧸2025", "Caracteres especiais"),
        ("Esta é uma resposta normal que deve ser falada.", "Normal"),
    ]
    
    for texto, label in test_cases:
        filtrado = filter.filter(texto)
        should_skip = filter.should_skip_sentence(texto)
        
        print(f"[{label}]")
        print(f"  Original: {texto}")
        print(f"  Filtrado: {filtrado}")
        print(f"  Skip? {'❌ SIM' if should_skip else '✅ NÃO'}")
        print()
    
    print("✓ Teste de filtros concluído\n")


def test_brain_response():
    """Testa resposta do brain (sem TTS para ser rápido)"""
    print("\n" + "="*60)
    print("TESTE 2: Brain Response (sem TTS)")
    print("="*60 + "\n")
    
    try:
        from brain_streaming import RenanBrainStreaming
        
        # Cria brain SEM TTS para teste rápido
        brain = RenanBrainStreaming(enable_tts=False)
        
        print("Pergunta: Qual sua visão sobre política?\n")
        
        # Faz pergunta
        response = brain.chat("Qual sua visão sobre política?", speak=False)
        
        # Valida resposta
        print(f"\n📊 Validação:")
        print(f"  Tamanho: {len(response)} caracteres")
        print(f"  Contém URL? {'❌ SIM' if 'http' in response.lower() else '✅ NÃO'}")
        print(f"  Contém 'Fonte'? {'❌ SIM' if 'fonte' in response.lower() else '✅ NÃO'}")
        print(f"  Contém 'Link:'? {'❌ SIM' if 'link:' in response.lower() else '✅ NÃO'}")
        
        if 'http' not in response.lower() and 'fonte' not in response.lower():
            print("\n✅ Resposta limpa! Brain funcionando corretamente.")
        else:
            print("\n⚠️ Resposta contém metadados. Revisar filtros.")
        
    except Exception as e:
        print(f"❌ Erro: {e}")


def test_brain_with_tts():
    """Testa brain com TTS (pergunta curta)"""
    print("\n" + "="*60)
    print("TESTE 3: Brain COM TTS (pergunta curta)")
    print("="*60 + "\n")
    
    try:
        from brain_streaming import RenanBrainStreaming
        
        print("⚠️  Este teste VAI FALAR (certifique-se de que o áudio está configurado)\n")
        
        continuar = input("Continuar? (s/N): ").strip().lower()
        
        if continuar != 's':
            print("Teste pulado.")
            return
        
        # Cria brain COM TTS
        brain = RenanBrainStreaming(
            enable_tts=True,
            tts_monitor=True,
            tts_speed=1.3  # Mais rápido para teste
        )
        
        # Pergunta curta
        print("\nPergunta: Olá, quem é você?\n")
        response = brain.chat("Olá, quem é você?")
        
        print("\n✅ Teste com TTS concluído!")
        print("\nVocê deveria ter ouvido:")
        print("  ✓ Apenas a resposta do Renan")
        print("  ✗ NENHUMA URL")
        print("  ✗ NENHUM 'Fonte 1', 'Link:', etc.")
        
    except Exception as e:
        print(f"❌ Erro: {e}")


def test_rag_question():
    """Testa pergunta que aciona RAG"""
    print("\n" + "="*60)
    print("TESTE 4: Pergunta com RAG (sem TTS)")
    print("="*60 + "\n")
    
    try:
        from brain_streaming import RenanBrainStreaming
        
        brain = RenanBrainStreaming(enable_tts=False)
        
        print("Pergunta: O que é direito penal do inimigo?\n")
        print("(Esta pergunta deve acionar o RAG)\n")
        
        response = brain.chat("O que é direito penal do inimigo?", speak=False)
        
        print(f"\n📊 Validação:")
        print(f"  RAG foi acionado? {'✅ SIM' if '[DEBUG RAG]' in str(response) else 'Confira logs acima'}")
        print(f"  Resposta limpa? {'✅ SIM' if 'http' not in response.lower() else '❌ NÃO'}")
        print(f"  Tamanho: {len(response)} chars")
        
        if 'http' not in response.lower() and 'fonte' not in response.lower():
            print("\n✅ RAG acionado + resposta limpa!")
        else:
            print("\n⚠️ Resposta contém URLs/fontes. Revisar.")
        
    except Exception as e:
        print(f"❌ Erro: {e}")


def menu():
    """Menu interativo"""
    print("""
╔══════════════════════════════════════════════════════════╗
║                                                          ║
║         🧪 TESTE RÁPIDO - CORREÇÕES TTS                 ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝

Escolha um teste:

1. Testar filtros de texto (rápido)
2. Testar brain sem TTS (médio)
3. Testar brain COM TTS - áudio (lento)
4. Testar pergunta com RAG (médio)
5. Executar todos os testes

0. Sair
    """)
    
    escolha = input("➤ Escolha (0-5): ").strip()
    
    if escolha == "1":
        test_filters()
    
    elif escolha == "2":
        test_brain_response()
    
    elif escolha == "3":
        test_brain_with_tts()
    
    elif escolha == "4":
        test_rag_question()
    
    elif escolha == "5":
        print("\n🚀 Executando todos os testes...\n")
        test_filters()
        input("\n⏸️  Pressione ENTER para continuar...")
        test_brain_response()
        input("\n⏸️  Pressione ENTER para continuar...")
        test_rag_question()
        input("\n⏸️  Pressione ENTER para teste com TTS...")
        test_brain_with_tts()
        print("\n✅ Todos os testes concluídos!")
    
    elif escolha == "0":
        print("\n👋 Até logo!")
    
    else:
        print("\n❌ Opção inválida")


if __name__ == "__main__":
    import sys
    
    # Comandos diretos
    if len(sys.argv) > 1:
        cmd = sys.argv[1].lower()
        
        if cmd == "filters":
            test_filters()
        elif cmd == "brain":
            test_brain_response()
        elif cmd == "tts":
            test_brain_with_tts()
        elif cmd == "rag":
            test_rag_question()
        elif cmd == "all":
            test_filters()
            test_brain_response()
            test_rag_question()
            test_brain_with_tts()
        else:
            print(f"Comando desconhecido: {cmd}")
            print("\nUso:")
            print("  python quick_test.py filters  # Testa filtros")
            print("  python quick_test.py brain    # Testa brain")
            print("  python quick_test.py tts      # Testa com TTS")
            print("  python quick_test.py rag      # Testa RAG")
            print("  python quick_test.py all      # Todos os testes")
    else:
        # Menu interativo
        menu()