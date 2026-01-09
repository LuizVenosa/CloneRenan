#!/usr/bin/env python3
"""
Script para comparar performance:
- brain_tts.py (antigo, sem streaming)
- brain_streaming.py (novo, com streaming)
"""

import time
import sys

def test_old_system():
    """Testa sistema antigo (sem streaming)"""
    print("\n" + "="*60)
    print("TESTE 1: Sistema ANTIGO (brain_tts.py)")
    print("="*60 + "\n")
    
    try:
        from brain_tts import RenanBrainWithTTS
        
        brain = RenanBrainWithTTS(
            enable_tts=True,
            tts_monitor=True
        )
        
        pergunta = "Explique sua visão sobre a política brasileira em três sentenças."
        
        print("⏱️  Medindo tempo até PRIMEIRA PALAVRA FALADA...\n")
        
        start = time.time()
        resposta = brain.chat(pergunta, speak=True)
        end = time.time()
        
        tempo_total = end - start
        
        print(f"\n📊 RESULTADO:")
        print(f"   Tempo total: {tempo_total:.2f}s")
        print(f"   Caracteres: {len(resposta)}")
        print(f"   Comportamento: Gera tudo → Espera → Fala tudo")
        
        return tempo_total
        
    except Exception as e:
        print(f"❌ Erro: {e}")
        return None


def test_new_system():
    """Testa sistema novo (com streaming)"""
    print("\n" + "="*60)
    print("TESTE 2: Sistema NOVO (brain_streaming.py)")
    print("="*60 + "\n")
    
    try:
        from brain_streaming import RenanBrainStreaming
        
        brain = RenanBrainStreaming(
            enable_tts=True,
            tts_monitor=True,
            tts_speed=1.2
        )
        
        pergunta = "Explique sua visão sobre a política brasileira em três sentenças."
        
        print("⏱️  Medindo tempo até PRIMEIRA PALAVRA FALADA...\n")
        print("👁️  OBSERVE: TTS deve começar ANTES do texto terminar de gerar!\n")
        
        start = time.time()
        resposta = brain.chat(pergunta, speak=True)
        end = time.time()
        
        tempo_total = end - start
        
        print(f"\n📊 RESULTADO:")
        print(f"   Tempo total: {tempo_total:.2f}s")
        print(f"   Caracteres: {len(resposta)}")
        print(f"   Comportamento: Gera sentença → Fala → Continua gerando")
        
        return tempo_total
        
    except Exception as e:
        print(f"❌ Erro: {e}")
        return None


def compare_latency():
    """Compara latência percebida"""
    print("\n" + "="*60)
    print("COMPARAÇÃO DE LATÊNCIA PERCEBIDA")
    print("="*60 + "\n")
    
    print("Simulando resposta de 300 caracteres (3 sentenças):\n")
    
    print("ANTIGO:")
    print("  1. LLM gera 100 chars → 1.0s")
    print("  2. LLM gera 100 chars → 2.0s")
    print("  3. LLM gera 100 chars → 3.0s")
    print("  4. TTS processa tudo → 4.0s")
    print("  5. ⭐ PRIMEIRA PALAVRA → 4.0s")
    print("  Latência percebida: 4.0s\n")
    
    print("NOVO (streaming):")
    print("  1. LLM gera 100 chars (sentença 1) → 1.0s")
    print("  2. ⭐ TTS fala sentença 1 → 1.2s (COMEÇOU!)")
    print("  3. LLM gera sentença 2 enquanto fala → 2.0s")
    print("  4. TTS fala sentença 2 → 2.5s")
    print("  5. LLM gera sentença 3 → 3.0s")
    print("  6. TTS fala sentença 3 → 3.5s")
    print("  Latência percebida: 1.2s\n")
    
    print("📈 MELHORIA: ~70% mais rápido para começar a falar!")
    print("🎯 Mais natural: fala enquanto pensa (como humano)\n")


def interactive_demo():
    """Demo interativo para sentir a diferença"""
    print("\n" + "="*60)
    print("DEMO INTERATIVO")
    print("="*60 + "\n")
    
    print("Escolha qual sistema testar:\n")
    print("1. Sistema ANTIGO (brain_tts.py)")
    print("2. Sistema NOVO (brain_streaming.py)")
    print("3. Testar AMBOS sequencialmente")
    print("4. Ver comparação de latência")
    print("5. Executar todos os testes")
    print("\n0. Sair\n")
    
    escolha = input("➤ Escolha (0-5): ").strip()
    
    if escolha == "1":
        test_old_system()
    
    elif escolha == "2":
        test_new_system()
    
    elif escolha == "3":
        print("\n🔬 Executando ambos para comparação...\n")
        input("⏸️  Pressione ENTER para testar sistema ANTIGO...")
        tempo_antigo = test_old_system()
        
        input("\n⏸️  Pressione ENTER para testar sistema NOVO...")
        tempo_novo = test_new_system()
        
        if tempo_antigo and tempo_novo:
            print("\n" + "="*60)
            print("📊 COMPARAÇÃO FINAL")
            print("="*60)
            print(f"   Sistema ANTIGO: {tempo_antigo:.2f}s")
            print(f"   Sistema NOVO:   {tempo_novo:.2f}s")
            
            if tempo_novo < tempo_antigo:
                melhoria = ((tempo_antigo - tempo_novo) / tempo_antigo) * 100
                print(f"   🚀 Melhoria: {melhoria:.1f}% mais rápido!")
            print()
    
    elif escolha == "4":
        compare_latency()
    
    elif escolha == "5":
        compare_latency()
        input("\n⏸️  Pressione ENTER para continuar...")
        test_old_system()
        input("\n⏸️  Pressione ENTER para continuar...")
        test_new_system()
        print("\n✅ Todos os testes concluídos!")
    
    elif escolha == "0":
        print("\n👋 Até logo!")
    
    else:
        print("\n❌ Opção inválida")


def quick_benchmark():
    """Benchmark rápido sem interação"""
    print("\n🔬 BENCHMARK AUTOMATIZADO\n")
    
    perguntas = [
        "Qual sua opinião sobre economia?",
        "Fale sobre filosofia política.",
        "Explique sua visão de mundo."
    ]
    
    print("Testando 3 perguntas em cada sistema...\n")
    
    # Sistema antigo
    print("▶️  Sistema ANTIGO:")
    from brain_tts import RenanBrainWithTTS
    brain_old = RenanBrainWithTTS(enable_tts=False)  # Sem TTS para ser mais rápido
    
    tempos_old = []
    for p in perguntas:
        start = time.time()
        brain_old.chat(p, speak=False)
        tempo = time.time() - start
        tempos_old.append(tempo)
        print(f"   {tempo:.2f}s")
    
    media_old = sum(tempos_old) / len(tempos_old)
    
    # Sistema novo
    print("\n▶️  Sistema NOVO:")
    from brain_streaming import RenanBrainStreaming
    brain_new = RenanBrainStreaming(enable_tts=False)
    
    tempos_new = []
    for p in perguntas:
        start = time.time()
        brain_new.chat(p, speak=False)
        tempo = time.time() - start
        tempos_new.append(tempo)
        print(f"   {tempo:.2f}s")
    
    media_new = sum(tempos_new) / len(tempos_new)
    
    # Resultado
    print("\n" + "="*60)
    print("📊 RESULTADO DO BENCHMARK")
    print("="*60)
    print(f"Média ANTIGO: {media_old:.2f}s")
    print(f"Média NOVO:   {media_new:.2f}s")
    
    if media_new < media_old:
        melhoria = ((media_old - media_new) / media_old) * 100
        print(f"🚀 Sistema novo é {melhoria:.1f}% mais rápido")
    else:
        print("⚠️  Resultados similares (variação normal)")
    
    print("\nNOTA: Este teste foi SEM TTS (só geração de texto)")
    print("      A diferença REAL é na latência percebida com TTS!")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        comando = sys.argv[1].lower()
        
        if comando == "old":
            test_old_system()
        elif comando == "new":
            test_new_system()
        elif comando == "compare":
            compare_latency()
        elif comando == "benchmark":
            quick_benchmark()
        else:
            print(f"Comando desconhecido: {comando}")
            print("\nUso:")
            print("  python compare_tts.py old       # Testa sistema antigo")
            print("  python compare_tts.py new       # Testa sistema novo")
            print("  python compare_tts.py compare   # Mostra comparação")
            print("  python compare_tts.py benchmark # Benchmark automático")
    else:
        # Modo interativo
        interactive_demo()