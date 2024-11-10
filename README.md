<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body>

<h1>Projeto de Análise e Classificação de Flores 🌸</h1>

<p>Este projeto utiliza o dataset de flores (Iris) para realizar uma análise exploratória e treinar modelos de classificação usando algoritmos de Machine Learning. O objetivo é prever a espécie de flor com base em características como comprimento e largura das pétalas e sépalas.</p>

<h2>Estrutura do Projeto</h2>

<ul>
    <li><strong>Carregamento de Dados</strong>: Carrega o dataset Iris a partir de um arquivo CSV com as colunas <code>SepalLengthCm</code>, <code>SepalWidthCm</code>, <code>PetalLengthCm</code>, <code>PetalWidthCm</code> e <code>Species</code>.</li>
    <li><strong>Pré-processamento</strong>: Realiza tratamento de dados para lidar com valores ausentes, normalização e visualização de outliers, além de mapear as classes de <code>Species</code> para valores numéricos.</li>
    <li><strong>Visualização</strong>: Gera gráficos de distribuição das variáveis e visualização de dispersão entre características das flores.</li>
    <li><strong>Modelagem</strong>: Treinamento de modelos de classificação utilizando <strong>SVM</strong> e <strong>KNN</strong> para prever a espécie de flor. A normalização dos dados é realizada usando o <code>MinMaxScaler</code>.</li>
    <li><strong>Avaliação</strong>: Avaliação do desempenho dos modelos com métricas como acurácia, precisão, recall, F1-score e matriz de confusão.</li>
</ul>

<h2>Dependências</h2>

<p>Para executar este projeto, são necessárias as seguintes bibliotecas Python:</p>
<ul>
    <li>pandas</li>
    <li>numpy</li>
    <li>scikit-learn</li>
    <li>matplotlib</li>
</ul>

<h2>Resultados</h2>

<p>Os modelos são avaliados com métricas de desempenho e as matrizes de confusão são visualizadas para análise. O desempenho dos modelos (SVM e KNN) é comparado para determinar qual tem melhor precisão na classificação das espécies de flores.</p>

<h2>Visualizações</h2>

<p>O projeto inclui gráficos de distribuição das características e uma visualização dos dados de teste e predições usando matrizes de confusão para cada modelo treinado.</p>

<h2>Autores</h2>

<p>Desenvolvido por Rodrigo.</p>

</body>
</html>
