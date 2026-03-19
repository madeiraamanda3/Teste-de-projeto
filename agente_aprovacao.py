import unicodedata
import pandas as pd
from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic_ai.models.openrouter import OpenRouterModel

load_dotenv()

ARQUIVO_CSV = "precos_aprovados.csv"


def remover_acentos(texto: str) -> str:
    return unicodedata.normalize('NFKD', texto).encode('ASCII', 'ignore').decode('ASCII')


def consultar_aprovacao(busca: str, quantidade: int, preco_unitario: float) -> str:
    """
    Consulta o CSV de aprovações e verifica se o pedido está aprovado.
    Busca o produto por SKU exato ou por parte do nome do produto.
    Ignora acentos tanto na busca quanto nos dados da planilha.

    A solicitação só é APROVADA se AMBOS os critérios forem atendidos:
    - Quantidade solicitada >= quantidade mínima aprovada
    - Preço unitário ofertado >= preço mínimo aprovado

    Se reprovada, retorna uma contraproposta válida com os valores mínimos aceitos.

    Args:
        busca: SKU numérico ou parte do nome do produto (ex: 24486 ou 'azeite')
        quantidade: Quantidade solicitada no pedido
        preco_unitario: Preço unitário ofertado no pedido (em R$)
    """
    try:
        df = pd.read_csv(ARQUIVO_CSV, sep=';')
        df.columns = [str(c).strip() for c in df.columns]

        # Tenta buscar por SKU exato primeiro
        mask_sku = df.iloc[:, 0].astype(str).str.strip() == str(busca).strip()
        produto = df[mask_sku]

        # Se não achou por SKU, busca por nome parcial ignorando acentos
        if produto.empty:
            busca_sem_acento = remover_acentos(busca.upper().strip())
            mask_nome = df.iloc[:, 1].astype(str).apply(
                lambda x: busca_sem_acento in remover_acentos(x.upper())
            )
            produto = df[mask_nome]

            if produto.empty:
                return (
                    f"Nenhum produto encontrado com '{busca}'. "
                    "Verifique o SKU ou tente outra parte do nome."
                )

            # Se achou mais de um produto, lista as opções
            if len(produto) > 1:
                opcoes = []
                opcoes.append(f"Encontrei {len(produto)} produtos com '{busca}'. Qual deles?")
                opcoes.append("")
                for _, row in produto.iterrows():
                    opcoes.append(f"  SKU {row.iloc[0]} — {str(row.iloc[1]).strip()}")
                opcoes.append("")
                opcoes.append("Informe o SKU exato para continuar.")
                return "\n".join(opcoes)

        linha = produto.iloc[0]
        qtd_minima   = float(str(linha.iloc[2]).replace('R$', '').replace('.', '').replace(',', '.').strip())
        preco_minimo = float(str(linha.iloc[3]).replace('R$', '').replace('.', '').replace(',', '.').strip())
        descricao    = str(linha.iloc[1]).strip()
        sku          = str(linha.iloc[0]).strip()

        qtd_ok   = quantidade >= qtd_minima
        preco_ok = preco_unitario >= preco_minimo
        aprovado = qtd_ok and preco_ok

        linhas = []
        linhas.append(f"Produto : {sku} — {descricao}")
        linhas.append(f"Status  : {'✅ APROVADO' if aprovado else '❌ REPROVADO'}")
        linhas.append("")
        linhas.append("ANÁLISE DA SOLICITAÇÃO:")
        linhas.append(f"  Quantidade  — Solicitada: {quantidade} | Mínima: {int(qtd_minima)} → {'✅' if qtd_ok else '❌'}")
        linhas.append(f"  Preço unit. — Ofertado: R$ {preco_unitario:.2f} | Mínimo: R$ {preco_minimo:.2f} → {'✅' if preco_ok else '❌'}")

        if not aprovado:
            qtd_proposta   = int(qtd_minima) if not qtd_ok else quantidade
            preco_proposta = round(preco_minimo, 2) if not preco_ok else preco_unitario
            linhas.append("")
            linhas.append("CONTRAPROPOSTA VÁLIDA:")
            linhas.append(f"  Quantidade  : {qtd_proposta}" + (" ← ajustado" if not qtd_ok else ""))
            linhas.append(f"  Preço unit. : R$ {preco_proposta:.2f}" + (" ← ajustado" if not preco_ok else ""))
            linhas.append(f"  Total       : R$ {qtd_proposta * preco_proposta:.2f}")

        return "\n".join(linhas)

    except FileNotFoundError:
        return f"Arquivo '{ARQUIVO_CSV}' não encontrado. Verifique o caminho configurado."
    except Exception as e:
        return f"Erro ao consultar o arquivo: {e}"


# ── Configuração do agente ────────────────────────────────────────────────────

modelo = OpenRouterModel("openai/gpt-4o-mini")

agente = Agent(
    model=modelo,
    tools=[consultar_aprovacao],
    system_prompt="""Você é um assistente de aprovação de pedidos de compra.
Quando receber uma solicitação com SKU ou nome do produto, quantidade e preço,
SEMPRE use a ferramenta consultar_aprovacao antes de responder — nunca responda de memória.
A solicitação só é aprovada se quantidade E preço estiverem dentro dos critérios.

Se REPROVADA, siga exatamente este fluxo:
1. Responda de forma objetiva e simpática: 'Infelizmente não consigo aprovar essa proposta. Consigo chegar em R$ [preço contraproposta] para [quantidade contraproposta] unidades. Funciona?'
2. Se o usuário CONCORDAR com a contraproposta (responder sim, ok, aceito, pode ser, etc), siga para o fluxo de aprovação: pergunte a UNB e o código do PDV, monte o resumo e aguarde confirmação.
3. Se o usuário DISCORDAR ou recusar a contraproposta, responda exatamente: 'Entendido! Vou te encaminhar para negociação com o nosso time. Em breve alguém entrará em contato!' e encerre sem perguntar mais nada.

Se APROVADA, siga exatamente este fluxo:
1. Informe que a proposta foi aprovada.
2. Pergunte: 'Qual a UNB e o código do PDV?'
3. Quando o usuário informar a UNB e o PDV, guarde essas informações e apresente o resumo completo no formato:
📋 RESUMO DO PEDIDO
  SKU        : [codigo]
  Produto    : [nome]
  Quantidade : [quantidade]
  Preço unit.: R$ [preco]
  UNB        : [unb]
  PDV        : [pdv]

Confirma?
4. Se o usuário CONFIRMAR, responda exatamente: '✅ O preço será cadastrado no Cora e te avisarei quando estiver tudo certo! Deseja negociar mais algum item?'
5. Se o usuário NÃO CONFIRMAR o resumo, pergunte de forma simpática o que está errado, corrija a informação indicada, apresente o resumo atualizado novamente no mesmo formato e pergunte 'Confirma?' Repita este passo quantas vezes for necessário até o usuário confirmar.

Se houver múltiplos produtos encontrados, apresente as opções ao usuário e peça para escolher.
Mantenha o contexto de toda a conversa para entender respostas curtas como 'sim', 'não', 'ok'.""",
)


# ── Loop de conversa interativo ───────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 65)
    print("  ASSISTENTE DE APROVAÇÃO DE PEDIDOS")
    print("=" * 65)
    print("\nOlá! Informe o SKU ou nome do produto, a quantidade e o")
    print("preço unitário que deseja negociar.")
    print("(digite 'sair' para encerrar)\n")

    historico = []

    while True:
        entrada = input("Você: ").strip()

        if entrada.lower() in ("sair", "exit", "quit"):
            print("\nAté logo!")
            break

        if not entrada:
            continue

        try:
            resultado = agente.run_sync(entrada, message_history=historico)
            resposta = resultado.output
            historico = resultado.all_messages()
            print(f"\nAssistente:\n{resposta}\n")
        except Exception as e:
            print(f"\n❌ ERRO REAL: {e}\n")
        print("-" * 65)