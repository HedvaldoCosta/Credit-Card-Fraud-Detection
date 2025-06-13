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

### Lidando com o desbalanceamento
Durante os testes, foram avaliados quatro abordagens para lidar com o desbalanceamento extremo do conjunto de dados (fraudes ≈ 0,17%):

## Comparação de Técnicas de Balanceamento

| Técnica           |   TP |   FN |    FP |   Precision |   Recall |         F1 |   ROC AUC |   Threshold |
|:------------------|-----:|-----:|------:|------------:|---------:|-----------:|----------:|------------:|
| **Sem Balanceamento** |  119 |   29 |    34 |  0.777778   | 0.804054 | 0.790698   |  0.952014 |        0.41 |
| NearMiss-1        |  144 |    4 | 74666 |  0.00192488 | 0.972973 | 0.00384215 |  0.902703 |        0.5  |
| NearMiss-2        |  128 |   20 | 29475 |  0.00432389 | 0.864865 | 0.00860475 |  0.881739 |        0.5  |
| NearMiss-3        |  118 |   30 |   170 |  0.409722   | 0.797297 | 0.541284   |  0.912913 |        0.5  |
| ADASYN            |  122 |   26 |  2167 |  0.0532984  | 0.824324 | 0.100123   |  0.94517  |        0.5  |
| SMOTE+Tomek       |  119 |   29 |   422 |  0.219963   | 0.804054 | 0.345428   |  0.950103 |        0.5  |
| EasyEnsemble      |  126 |   22 |  3661 |  0.0332717  | 0.851351 | 0.0640407  |  0.943477 |        0.5  |
| BalancedRF        |  130 |   18 |  2618 |  0.0473071  | 0.878378 | 0.089779   |  0.958013 |        0.5  |

Foi concluído que modelos robustos que conseguem lidar com grande desbalanceamento, igual ao `XGBoost` e `Random Forest`, conseguiram melhores resultados utilizando o conjunto desbalanceado do que com técnicas de balanceamento.

## 🤖 Escolha do Modelo Final
**Modelos comparados sem reamostragem:**
| Modelo            | Precision  | Recall     | F1 Score   | ROC AUC    |
| ----------------- | ---------- | ---------- | ---------- | ---------- |
| **Random Forest** | **0.9397** | 0.7826     | **0.8537** | 0.9297     |
| **XGBoost**       | 0.7999     | **0.8252** | 0.8122     | **0.9554** |

Foi optado pelo ``Random Forest`` como modelo final, por apresentar o melhor equilíbrio entre precisão e capacidade geral do modelo, especialmente considerando o objetivo de evitar falsos alarmes excessivos.

### Treinando o modelo
O modelo ``Random Forest`` foi treinado com 70% do conjunto de dados e 30% para o teste do modelo. Inicialmente, o modelo foi treinado com threshold de 0.5 e forneceu os seguintes resultados:

```python
features = ['V14', 'V10', 'V12', 'V17', 'V11', 'V16']

X = df[features]
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

rf_model = RandomForestClassifier(class_weight='balanced', random_state=42)

rf_model.fit(X_train, y_train)
```

| Classe       | Métrica                   | Valor   |
| ------------ | ------------------------- | ------- |
| 0 (legítima) | TN (corretas)             | 85.290  |
| 0 (legítima) | FP (barradas por engano)  | **5**  |
| 1 (fraude)   | FN (fraudes que passaram) | **36**  |
| 1 (fraude)   | TP (fraudes detectadas)   | **112** |

Ou seja, o modelo teve um grande resultado em relação a falsos positivos, mas teve um erro de 24% em barrar fraudes legítimas.

Com isso, foram testados novos treinamentos com diferentes thresholds:

**threshold = 0.25**

| Classe       | Métrica                   | Valor   |
| ------------ | ------------------------- | ------- |
| 0 (legítima) | TN (corretas)             | 85.277  |
| 0 (legítima) | FP (barradas por engano)  | **18**  |
| 1 (fraude)   | FN (fraudes que passaram) | **30**  |
| 1 (fraude)   | TP (fraudes detectadas)   | **118** |

**Redução de 16,67% no número de fraudes, mas um aumento de 260% nas transações barradas por engano.**

**threshold = 0.20**

| Classe       | Métrica                   | Valor   |
| ------------ | ------------------------- | ------- |
| 0 (legítima) | TN (corretas)             | 85.273  |
| 0 (legítima) | FP (barradas por engano)  | **22**  |
| 1 (fraude)   | FN (fraudes que passaram) | **28**  |
| 1 (fraude)   | TP (fraudes detectadas)   | **120** |

**Redução de 22,22% no número de fraudes, mas um aumento de 340% nas transações barradas por engano.**

**threshold = 0.15**

| Classe       | Métrica                   | Valor   |
| ------------ | ------------------------- | ------- |
| 0 (legítima) | TN (corretas)             | 85.265  |
| 0 (legítima) | FP (barradas por engano)  | **30**  |
| 1 (fraude)   | FN (fraudes que passaram) | **27**  |
| 1 (fraude)   | TP (fraudes detectadas)   | **121** |

**Redução de 25% no número de fraudes, mas um aumento de 500% nas transações barradas por engano.**

---
**Identificando novas melhorias, foi testado o uso da calibração e funções para identificar a melhor combinação de parâmetros.**

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

def build_optimized_model(X_train, y_train):
    base_rf = RandomForestClassifier(
        class_weight='balanced',
        random_state=42
    )
    
    calibrated = CalibratedClassifierCV(
        base_rf,
        method='isotonic',
        cv=3
    )
    
    param_grid = {
        'base_estimator__n_estimators': [100, 200],
        'base_estimator__max_depth': [5, 10, None],
        'base_estimator__min_samples_leaf': [1, 2],
        'method': ['sigmoid', 'isotonic']
    }
    
    search = GridSearchCV(
        calibrated,
        param_grid,
        scoring='f1',
        cv=3,
        n_jobs=-1,
        verbose=1
    )
    
    search.fit(X_train, y_train)
    return search.best_estimator_

optimized_model = build_optimized_model(X_train, y_train)
y_proba_opt = optimized_model.predict_proba(X_test)[:, 1]
optimal_threshold_opt = find_optimal_threshold(y_test, y_proba_opt)
y_pred_opt = (y_proba_opt >= optimal_threshold_opt).astype(int)
```

**Onde os resultados foram minimamente melhores:**

| Classe       | Métrica                   | Valor   |
| ------------ | ------------------------- | ------- |
| 0 (legítima) | TN (corretas)             | 85.252  |
| 0 (legítima) | FP (barradas por engano)  | **43**  |
| 1 (fraude)   | FN (fraudes que passaram) | **25**  |
| 1 (fraude)   | TP (fraudes detectadas)   | **123** |
