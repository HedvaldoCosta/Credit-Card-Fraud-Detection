# 🛡️ Detecção de Fraude em Cartões de Crédito com Machine Learning

Este projeto apresenta uma abordagem de machine learning para identificar transações potencialmente fraudulentas em um grande conjunto de dados de cartões de crédito. O objetivo é desenvolver um modelo eficiente, confiável e interpretável para apoiar decisões em ambientes de risco financeiro.

---

## 📂 Sumário

- [Contexto do Problema](#contexto-do-problema)
- [Objetivos do Projeto](#objetivos-do-projeto)
- [Descrição dos Dados](#descrição-dos-dados)
- [Análise exploratória de dados](#analise-exploratoria-de-dados)
---

## 📌 Contexto do Problema

Fraudes com cartões de crédito causam bilhões em prejuízo às instituições financeiras anualmente. A maioria dos sistemas de detecção tradicionais apresenta dificuldades em:
- Adaptar-se a padrões de fraude em constante mudança
- Lidar com dados extremamente desbalanceados
- Minimizar falsos positivos sem deixar fraudes passarem despercebidas

---

## 🎯 Objetivos do Projeto

- Desenvolver um modelo de classificação binária para prever fraudes em transações de cartão de crédito
- Avaliar a performance com métricas adequadas a datasets desbalanceados
- Demonstrar um pipeline completo de machine learning com boas práticas de engenharia de dados e validação

---

## 🧾 Descrição dos Dados

- **Fonte**: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Tipos de dados**: 30 float64 e 1 int64
- **Total de transações**: 284.807 (sem dados ausentes)
- **Transações fraudulentas**: 492 (aproximadamente 0,172%)
- **Colunas principais**:
  - `V1` a `V28`: variáveis transformadas via PCA para anonimização
  - `Amount`: valor da transação
  - `Time`: segundos desde a primeira transação no dataset
  - `Class`: variável alvo (0 = legítima, 1 = fraude)

---
## 📊 Análise exploratória de dados

### Distribuição de classes
A variável alvo `Class` é uma variável binária (composta apenas por 1 e 0) assimétrica (as classes não estão distribuídas igualmente), com a classe de fraudes `1` representando uma pequena fração do total (**0,172%**). Ou seja, um modelo treinado com esses dados tenderia a prever sempre a classe majoritária, obtendo alta acurácia, mas ignorando as fraudes. Para isso, serão testadas as técnicas `Undersampling`, `Oversampling ` e `SMOTE `.

### Correlação entre variáveis
Foi testado, primeiramente, a correlação de Pearson. Mas, como se tratam de dados não lineares, todas as correlação foram baixas. Para isso, foi utilizado MI (Mutual Information) e RF (Random Forest), onde quanto maior o MI, mais a variável ajuda a reduzir a incerteza sobre a variável alvo `Class`.
O MI foi normalizado em relação à entropia da variável Class (`~0.01834 bits`) para calcular a % de entropia explicada. Após isso, foi treinada uma RF para extrair a importância atribuída a cada variável com base na redução de impureza. Para a seleção de variáveis, foram utilizados os seguintes critérios:
* MI ≥ 0.005
* % Entropia Explicada ≥ 20%
* RF Importance ≥ 0.03

| Variável | Mutual Information | % Entropia Explicada | Random Forest Importance |
|----------|--------------------|----------------------|---------------------------|
| V14      | 0.008136           | 44.35%               | 0.1772                    |
| V10      | 0.007530           | 41.05%               | 0.1163                    |
| V12      | 0.007601           | 41.44%               | 0.1057                    |
| V17      | 0.008258           | 45.02%               | 0.0888                    |
| V11      | 0.006831           | 37.24%               | 0.0520                    |
| V16      | 0.006144           | 33.50%               | 0.0498                    |


