# 🛡️ Detecção de Fraude em Cartões de Crédito com Machine Learning

Este projeto apresenta uma abordagem de machine learning para identificar transações potencialmente fraudulentas em um grande conjunto de dados de cartões de crédito. O objetivo é desenvolver um modelo eficiente, confiável e interpretável para apoiar decisões em ambientes de risco financeiro.

---

## 📂 Sumário

- [Contexto do Problema](#contexto-do-problema)
- [Objetivos do Projeto](#objetivos-do-projeto)
- [Descrição dos Dados](#descrição-dos-dados)

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
- **Total de transações**: 284.807
- **Transações fraudulentas**: 492 (aproximadamente 0,172%)
- **Colunas principais**:
  - `V1` a `V28`: variáveis transformadas via PCA para anonimização
  - `Amount`: valor da transação
  - `Time`: segundos desde a primeira transação no dataset
  - `Class`: variável alvo (0 = legítima, 1 = fraude)

---


