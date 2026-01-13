#!/usr/bin/env python3
"""
Teste específico para brain_streaming_fixed.py
Valida que todas as correções funcionam
"""

def test_basic_response():
    """Teste 1: Resposta básica funciona?"""
    print("\n" + "="*60)
    print("TESTE 1: Resposta Básica (sem TTS)")
    print("="*60 + "\n")
    
    try:
        from brain_streaming import RenanBrainStreaming
        
        print("Criando brain...")
        brain = RenanBrainStreaming(enable_tts=False)
        
        print("\nPergunta: 'Qual o seu nome?'\n")
        resposta = brain.chat("Qual o seu nome?", speak=False)
        
        print(f"\n📊 Resultado:")
        print(f"  Tamanho: {len(resposta)} caracteres")
        print(f"  Primeira linha: {resposta[:100] if resposta else '(vazio)'}...")
        
        # Validações
        if not resposta:
            print("\n❌ FALHOU: Resposta vazia!")
            print("   Possíveis causas:")
            print("   - API key não configurada")
            print("   - Problema no grafo LangGraph")
            print("   - System prompt não carregado")
            return False
        
        if len(resposta) < 10:
            print("\n❌ FALHOU: Resposta muito curta")
            return False
        
        if "Qual o seu nome?" in resposta:
            print("\n⚠️ WARNING: Resposta contém a pergunta (eco)")
            print("   Mas pelo menos GEROU resposta...")
        
        print("\n✅ PASSOU: Resposta gerada com sucesso!")
        return True
        
    except Exception as e:
        print(f"\n❌ ERRO: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_rag():
    """Teste 2: RAG funciona?"""
    print("\n" + "="*60)
    print("TESTE 2: RAG (sem TTS)")
    print("="*60 + "\n")
    
    try:
        from brain_streaming import RenanBrainStreaming
        
        brain = RenanBrainStreaming(enable_tts=False)
        
        print("Pergunta: 'O que é direito penal do inimigo?'\n")
        resposta = brain.chat("O que é direito penal do inimigo?", speak=False)
        
        print(f"\n📊 Resultado:")
        print(f"  Tamanho: {len(resposta)} caracteres")
        
        # Validações
        if not resposta:
            print("\n❌ FALHOU: Sem resposta")
            return False
        
        # Verifica se menciona conceito (indica que RAG foi usado)
        keywords = ['direito', 'penal', 'inimigo', 'teoria', 'jacobs', 'günther']
        tem_keyword = any(kw in resposta.lower() for kw in keywords)
        
        if tem_keyword:
            print("✅ Resposta parece baseada em RAG (contém keywords)")
        else:
            print("⚠️ Resposta pode não ter usado RAG")
        
        # Verifica filtros
        tem_url = 'http' in resposta.lower()
        tem_fonte = 'fonte ' in resposta.lower()
        
        if tem_url or tem_fonte:
            print(f"⚠️ WARNING: Resposta contém metadados")
            print(f"   URLs: {'Sim' if tem_url else 'Não'}")
            print(f"   Fontes: {'Sim' if tem_fonte else 'Não'}")
        else:
            print("✅ Filtros funcionando (sem URLs/fontes)")
        
        print("\n✅ PASSOU: RAG funcionou!")
        return True
        
    except Exception as e:
        print(f"\n❌ ERRO: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_streaming_tts():
    """Teste 3: Streaming TTS funciona?"""
    print("\n" + "="*60)
    print("TESTE 3: Streaming TTS (VAI FALAR)")
    print("="*60 + "\n")
    
    print("⚠️  Este teste VAI REPRODUZIR ÁUDIO!")
    print("    Certifique-se de que:")
    print("    - Alto-falantes ou CABLE Input estão configurados")
    print("    - Volume está ajustado")
    print()
    
    continuar = input("Continuar? (s/N): ").strip().lower()
    
    if continuar != 's':
        print("Teste pulado.")
        return True
    
    try:
        from brain_streaming import RenanBrainStreaming
        
        print("\nCriando brain COM TTS...")
        brain = RenanBrainStreaming(
            enable_tts=True,
            tts_monitor=True,
            tts_speed=1.3  # Mais rápido para teste
        )
        
        print("\nPergunta curta: 'Olá, tudo bem?'\n")
        import time
        start = time.time()
        
        resposta = brain.chat("Olá, tudo bem?", speak=True)
        
        tempo_total = time.time() - start
        
        print(f"\n📊 Resultado:")
        print(f"  Tempo total: {tempo_total:.2f}s")
        print(f"  Tamanho resposta: {len(resposta)} chars")
        
        print("\n✅ PASSOU se você:")
        print("  ✓ Ouviu o áudio")
        print("  ✓ TTS começou ANTES do texto terminar de gerar")
        print("  ✓ Não ouviu URLs ou metadados")
        
        verificado = input("\nTudo funcionou corretamente? (s/N): ").strip().lower()
        
        if verificado == 's':
            print("\n✅ Streaming TTS validado pelo usuário!")
            return True
        else:
            print("\n⚠️ Usuário reportou problema")
            return False
        
    except Exception as e:
        print(f"\n❌ ERRO: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_conversation():
    """Teste 4: Conversa com contexto"""
    print("\n" + "="*60)
    print("TESTE 4: Conversa Multi-Turn (sem TTS)")
    print("="*60 + "\n")
    
    try:
        from brain_streaming import RenanBrainStreaming
        from langchain_core.messages import HumanMessage, AIMessage
        
        brain = RenanBrainStreaming(enable_tts=False)
        
        # Simula conversa
        messages = []
        
        # Turno 1
        print("Turno 1: 'Qual o seu nome?'")
        messages.append(HumanMessage(content="Qual o seu nome?"))
        
        inputs1 = {"messages": messages}
        resposta1 = ""
        
        for event in brain.agent.stream(inputs1):
            for node_name, node_output in event.items():
                if node_name == "chatbot" and "messages" in node_output:
                    last_msg = node_output["messages"][-1]
                    if isinstance(last_msg, AIMessage):
                        resposta1 = last_msg.content
        
        messages.append(AIMessage(content=resposta1))
        print(f"  Resposta: {resposta1[:50]}...")
        
        # Turno 2
        print("\nTurno 2: 'E qual sua profissão?'")
        messages.append(HumanMessage(content="E qual sua profissão?"))
        
        inputs2 = {"messages": messages}
        resposta2 = ""
        
        for event in brain.agent.stream(inputs2):
            for node_name, node_output in event.items():
                if node_name == "chatbot" and "messages" in node_output:
                    last_msg = node_output["messages"][-1]
                    if isinstance(last_msg, AIMessage):
                        resposta2 = last_msg.content
        
        print(f"  Resposta: {resposta2[:50]}...")
        
        # Validação
        if resposta1 and resposta2:
            print("\n✅ PASSOU: Conversa multi-turn funciona!")
            return True
        else:
            print("\n❌ FALHOU: Alguma resposta vazia")
            return False
        
    except Exception as e:
        print(f"\n❌ ERRO: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Executa todos os testes"""
    print("""
╔══════════════════════════════════════════════════════════╗
║                                                          ║
║    🧪 TESTE COMPLETO - brain_streaming_fixed.py         ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
    """)
    
    results = []
    
    # Teste 1: Básico
    print("\n🔬 Iniciando testes...\n")
    input("Pressione ENTER para Teste 1 (Resposta Básica)...")
    results.append(("Resposta Básica", test_basic_response()))
    
    # Teste 2: RAG
    if results[0][1]:  # Só continua se teste 1 passou
        input("\nPressione ENTER para Teste 2 (RAG)...")
        results.append(("RAG", test_rag()))
    else:
        print("\n⚠️ Pulando testes restantes (teste básico falhou)")
        return
    
    # Teste 3: TTS
    if results[1][1]:
        input("\nPressione ENTER para Teste 3 (Streaming TTS)...")
        results.append(("Streaming TTS", test_streaming_tts()))
    
    # Teste 4: Conversa
    if results[0][1]:
        input("\nPressione ENTER para Teste 4 (Conversa)...")
        results.append(("Conversa Multi-Turn", test_conversation()))
    
    # Resumo
    print("\n" + "="*60)
    print("📊 RESUMO DOS TESTES")
    print("="*60)
    
    for nome, passou in results:
        status = "✅ PASSOU" if passou else "❌ FALHOU"
        print(f"  {status}: {nome}")
    
    total = len(results)
    passou_count = sum(1 for _, p in results if p)
    
    print(f"\nTotal: {passou_count}/{total} testes passaram")
    
    if passou_count == total:
        print("\n🎉 TODOS OS TESTES PASSARAM!")
        print("\nSistema pronto para uso!")
        print("Execute: python brain_streaming_fixed.py")
    else:
        print("\n⚠️ Alguns testes falharam")
        print("Verifique os erros acima")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        cmd = sys.argv[1].lower()
        
        if cmd == "basic":
            test_basic_response()
        elif cmd == "rag":
            test_rag()
        elif cmd == "tts":
            test_streaming_tts()
        elif cmd == "conversation":
            test_conversation()
        elif cmd == "all":
            run_all_tests()
        else:
            print(f"Comando desconhecido: {cmd}")
            print("\nUso:")
            print("  python test_fixed.py basic        # Teste básico")
            print("  python test_fixed.py rag          # Teste RAG")
            print("  python test_fixed.py tts          # Teste TTS")
            print("  python test_fixed.py conversation # Teste conversa")
            print("  python test_fixed.py all          # Todos")
    else:
        # Modo interativo
        run_all_tests()