# Model Card — Instacart Recommender

**Versão do modelo:** 1 (registrada no MLflow Model Registry, stage: `Production`)
**Data:** Julho de 2026
**Framework:** PyTorch 2.13
**Tipo:** Rede neural híbrida (embeddings + MLP), classificação binária
**Repositório:** [github.com/tiagotff/FIAP_MLE10_TC2](https://github.com/tiagotff/FIAP_MLE10_TC2)
**Licença:** MIT (código); dataset sob licença própria do Instacart/Kaggle (ver [Dataset](#dataset))

## Sumário

Modelo de recomendação que estima a probabilidade de um usuário
recomprar um produto específico, com base no seu histórico de pedidos
no [Instacart Market Basket Analysis](https://www.kaggle.com/datasets/psparks/instacart-market-basket-analysis).
O sinal é usado para ranquear produtos a sugerir no momento da compra.

## Detalhes do modelo

| | |
|---|---|
| Arquitetura | Embeddings de `user_id` e `product_id` (dim. 32 cada) concatenados com 5 features tabulares, seguidos de MLP `[128, 64, 32] → 1` |
| Regularização | `BatchNorm1d` + `Dropout(0.3)` entre camadas |
| Função de perda | `BCEWithLogitsLoss` |
| Otimizador | Adam, `learning_rate=0.001` |
| Batch size | 131.072 |
| Early stopping | `patience=3`, melhora mínima de AUC de 0,001 por época |
| Seed | 42 (fixada em `torch`, `numpy`/`sklearn` via `random_state`) |
| Parâmetros treináveis | Dominados pelas tabelas de embedding: ~206 mil usuários × 32 + ~50 mil produtos × 32, mais a MLP |

## Uso pretendido

**Uso primário:** ranquear produtos candidatos a recomendação para um
usuário específico de um serviço de e-commerce de mercado, com base em
histórico de pedidos — por exemplo, para popular uma seção "compre de
novo" ou reordenar resultados de busca por afinidade de recompra.

**Usuários pretendidos:** times de engenharia/produto de e-commerce
construindo funcionalidades de recomendação; neste projeto, avaliação
acadêmica (FIAP MLE10 Tech Challenge Fase 02).

**Fora do escopo:** o modelo **não** foi desenhado nem validado para:
recomendação de produtos nunca vistos pelo usuário (cold-start de
produto), recomendação para usuários novos sem histórico (cold-start de
usuário — os embeddings só existem para os IDs vistos no treino),
decisões de precificação, ou qualquer uso fora do domínio de varejo de
mercado/e-commerce.

## Dataset

**Instacart Market Basket Analysis** (Kaggle, dataset público).

- ~3,4 milhões de pedidos, ~206 mil usuários, ~50 mil produtos
- ~32,4 milhões de linhas produto-pedido (`order_products__prior.csv`)
- Rótulo: `reordered` (1 = produto recomprado naquele pedido)
- Split treino/validação: 80/20, aleatório, `random_state=42`
- Features usadas: `purchase_count`, `days_since_last_order`,
  `order_hour_of_day`, `order_dow`, `basket_size` (ver
  `src/recommender/preprocessing/strategies.py`), além dos embeddings
  de `user_id`/`product_id`

## Performance

Métricas no conjunto de validação (6.486.898 exemplos), calculadas por
`src/recommender/pipeline/evaluate.py` e logadas no MLflow:

| Métrica | Modelo neural (híbrido) | Baseline (Regressão Logística) |
|---|---|---|
| AUC-ROC | **0,9045** | 0,8964 |
| Recall | **0,9876** | 0,7698 |
| Precision | 0,7879 | **0,8777** |
| F1-score | **0,8765** | 0,8202 |

O baseline usa **só as features tabulares** (sem embeddings de
usuário/produto) — a diferença de performance isola o ganho específico
de aprender representações latentes de usuário e produto.

**Leitura dos números:** o modelo neural tem recall muito mais alto —
captura quase todos os casos reais de recompra —, ao custo de precision
um pouco menor. Para um caso de uso de recomendação (mostrar produtos
candidatos, não tomar decisões automáticas irreversíveis), essa troca
costuma ser desejável: o custo de sugerir um produto que não será
recomprado é baixo, enquanto deixar de sugerir um que seria é a
oportunidade perdida mais relevante.

## Limitações

- **Dataset desatualizado**: o Instacart Market Basket Analysis é de
  ~2017. Padrões de compra podem ter mudado (novos produtos, hábitos
  pós-pandemia, inflação), e o modelo não captura nenhuma tendência
  posterior a essa janela.
- **Cold-start**: usuários e produtos que não apareceram no conjunto de
  treino não têm embedding aprendido; o modelo não tem mecanismo para
  lidar com eles além de um índice fora do vocabulário.
- **Escopo geográfico e de categoria**: dataset reflete o comportamento
  de compra de mercado nos EUA através de um único varejista
  (Instacart); não generaliza para outras regiões, culturas de compra,
  ou categorias de produto fora de mercado/mantimentos.
- **Ausência de contexto de sessão em tempo real**: o modelo usa
  agregados históricos (recência, frequência, tamanho médio de
  carrinho), não sinais de navegação em tempo real (o que o enunciado
  do desafio menciona como motivação original, mas o dataset disponível
  não contém).
- **Sem teste A/B ou validação online**: as métricas refletem apenas
  avaliação offline (holdout); não há garantia de que a mesma
  performance se traduza em métricas de negócio (conversão, receita)
  em produção.

## Vieses conhecidos

- **Viés de popularidade**: produtos comprados com mais frequência
  aparecem desproporcionalmente mais vezes no treino, então o modelo
  tende a favorecer produtos populares em detrimento de itens de nicho
  — um padrão comum em sistemas de recomendação baseados em histórico
  de interação, não corrigido neste projeto (ex.: via reponderação ou
  penalização de popularidade).
- **Viés de usuários ativos**: usuários com mais pedidos no histórico
  contribuem com mais exemplos de treino, então o modelo pode
  performar melhor para usuários frequentes do que para usuários
  ocasionais.
- **Ausência de atributos demográficos**: o dataset não contém idade,
  gênero, renda ou localização dos usuários, o que limita a análise de
  equidade entre grupos demográficos — não é possível auditar o modelo
  quanto a esse tipo de viés com os dados disponíveis.

## Considerações éticas

O modelo influencia quais produtos são sugeridos a um usuário, o que
pode reforçar padrões de compra existentes (viés de popularidade,
acima) em vez de promover descoberta de produtos novos. Recomenda-se
que qualquer uso em produção combine este modelo com mecanismos de
diversificação e monitoramento contínuo de métricas de negócio e
equidade, não usá-lo como única fonte de decisão.

## Como servir em produção

O modelo em `Production` no MLflow Model Registry é servido via uma API
FastAPI (`src/recommender/api/`), com endpoints de liveness/readiness,
predição individual e em lote, metadados e métricas operacionais — ver
[README.md, seção "API de inferência"](README.md#api-de-inferência). A
API pode ser implantada em nuvem via Cloud Run, carregando os artefatos
dinamicamente de um bucket GCS (ver
[README.md, seção "Deploy em nuvem"](README.md#deploy-em-nuvem-bônus)).

## Como reproduzir

Ver [`README.md`](README.md) — seções **Quickstart** e **Pipeline de
dados e treino (DVC)**.
