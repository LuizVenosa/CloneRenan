#!/usr/bin/env python3
"""
Script de teste para o sistema Renan AI + TTS
Demonstra diferentes modos de uso
"""

from brain_tts import RenanBrainWithTTS

def test_single_question():
    """Teste 1: Uma pergunta simples"""
    print("\n" + "="*60)
    print("TESTE 1: Pergunta única com TTS")
    print("="*60 + "\n")
    
    brain = RenanBrainWithTTS(
        enable_tts=True,
        tts_output_device="CABLE Input",
        tts_monitor=True
    )
    
    resposta = brain.chat("Qual sua opinião sobre a economia brasileira?")
    print(f"\n✓ Resposta obtida ({len(resposta)} caracteres)")


def test_without_tts():
    """Teste 2: Sem TTS (só texto)"""
    print("\n" + "="*60)
    print("TESTE 2: Modo texto puro (sem TTS)")
    print("="*60 + "\n")
    
    brain = RenanBrainWithTTS(enable_tts=False)
    
    perguntas = [
        "Quem é você?",
        "O que você pensa sobre política?",
        "Fale sobre filosofia"
    ]
    
    for p in perguntas:
        resposta = brain.chat(p)
        print()


def test_rag_triggered():
    """Teste 3: Pergunta que deve acionar RAG"""
    print("\n" + "="*60)
    print("TESTE 3: Acionando RAG + TTS")
    print("="*60 + "\n")
    
    brain = RenanBrainWithTTS(
        enable_tts=True,
        tts_monitor=True
    )
    
    # Pergunta específica que deve buscar na memória
    resposta = brain.chat(
        "O que você já falou sobre Bitcoin nas suas lives?"
    )
    
    print(f"\n✓ RAG deve ter sido acionado. Resposta: {len(resposta)} chars")


def test_interactive_session():
    """Teste 4: Sessão interativa completa"""
    print("\n" + "="*60)
    print("TESTE 4: Sessão interativa")
    print("="*60 + "\n")
    
    brain = RenanBrainWithTTS(
        enable_tts=True,
        tts_output_device="CABLE Input",
        tts_monitor=True
    )
    
    # Inicia sessão (loop até 'sair')
    brain.chat_session()


def test_controlled_speech():
    """Teste 5: Controle manual do TTS"""
    print("\n" + "="*60)
    print("TESTE 5: Controle manual de fala")
    print("="*60 + "\n")
    
    brain = RenanBrainWithTTS(enable_tts=False)  # TTS desligado por padrão
    
    # Primeira pergunta: não fala
    print("▶️ Sem TTS:")
    r1 = brain.chat("Olá, tudo bem?", speak=False)
    
    # Segunda pergunta: fala
    print("\n▶️ Com TTS:")
    r2 = brain.chat("Explique sua visão política", speak=True)
    
    # Terceira: volta a não falar
    print("\n▶️ Sem TTS novamente:")
    r3 = brain.chat("Obrigado", speak=False)


def demo_showcase():
    """Demo completa para apresentação"""
    print("""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║          🧠 RENAN SANTOS AI - SISTEMA COMPLETO 🎙️             ║
║                                                              ║
║  LLM (Gemini) + RAG (ChromaDB) + TTS (Edge-TTS)              ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    print("\nEscolha um teste:\n")
    print("1. Pergunta única com TTS")
    print("2. Modo texto puro (sem TTS)")
    print("3. Teste de RAG + TTS")
    print("4. Sessão interativa completa")
    print("5. Controle manual de TTS")
    print("6. Executar todos os testes")
    print("\n0. Sair")
    
    escolha = input("\n➤ Escolha (0-6): ").strip()
    
    if escolha == "1":
        test_single_question()
    elif escolha == "2":
        test_without_tts()
    elif escolha == "3":
        test_rag_triggered()
    elif escolha == "4":
        test_interactive_session()
    elif escolha == "5":
        test_controlled_speech()
    elif escolha == "6":
        print("\n🚀 Executando todos os testes...\n")
        test_single_question()
        input("\n⏸️  Pressione ENTER para próximo teste...")
        test_without_tts()
        input("\n⏸️  Pressione ENTER para próximo teste...")
        test_rag_triggered()
        input("\n⏸️  Pressione ENTER para próximo teste...")
        test_controlled_speech()
        print("\n✅ Todos os testes concluídos!")
    elif escolha == "0":
        print("\n👋 Até logo!")
    else:
        print("\n❌ Opção inválida")


# ============================================================================
# EXECUÇÃO PRINCIPAL
# ============================================================================

if __name__ == "__main__":
    import sys
    
    # Se rodar sem argumentos, mostra menu
    if len(sys.argv) == 1:
        demo_showcase()
    
    # Se rodar com argumento, executa modo direto
    else:
        comando = sys.argv[1].lower()
        
        if comando == "chat":
            # Modo chat interativo
            brain = RenanBrainWithTTS(
                enable_tts=True,
                tts_monitor=True
            )
            brain.chat_session()
        
        elif comando == "silent":
            # Modo silencioso (sem TTS)
            brain = RenanBrainWithTTS(enable_tts=False)
            brain.chat_session()
        
        elif comando == "test":
            # Testa uma pergunta e sai
            if len(sys.argv) < 3:
                print("Uso: python test_chat_tts.py test 'sua pergunta aqui'")
                sys.exit(1)
            
            pergunta = " ".join(sys.argv[2:])
            brain = RenanBrainWithTTS(enable_tts=True)
            brain.chat(pergunta)
        
        else:
            print(f"Comando desconhecido: {comando}")
            print("\nUso:")
            print("  python test_chat_tts.py          # Menu interativo")
            print("  python test_chat_tts.py chat     # Chat com TTS")
            print("  python test_chat_tts.py silent   # Chat sem TTS")
            print("  python test_chat_tts.py test 'pergunta'")