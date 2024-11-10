import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, roc_auc_score
import matplotlib.pyplot as plt

class Modelo():
    def __init__(self):
        self.inv_name_dict = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}

    def CarregarDataset(self, path):
        """
        Carrega o conjunto de dados a partir de um arquivo CSV.

        Parâmetros:
        - path (str): Caminho para o arquivo CSV contendo o dataset.
        
        O dataset é carregado com as seguintes colunas: SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm e Species.
        """
        names = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm', 'Species']
        self.df = pd.read_csv(path, names=names)
        

    def TratamentoDeDados(self):
        """
        Realiza o pré-processamento dos dados carregados.

        Sugestões para o tratamento dos dados:
            * Utilize `self.df.head()` para visualizar as primeiras linhas e entender a estrutura.
            * Verifique a presença de valores ausentes e faça o tratamento adequado.
            * Considere remover colunas ou linhas que não são úteis para o treinamento do modelo.
        
        Dicas adicionais:
            * Explore gráficos e visualizações para obter insights sobre a distribuição dos dados.
            * Certifique-se de que os dados estão limpos e prontos para serem usados no treinamento do modelo.
        """
        print(self.df.head())

        print(self.df.isnull().sum())
        print(self.df.groupby(['Species']).mean())


    def Visualizacoes(self):
        """
        Método que contém uma análise exploratória dos dados. Utilizando a biblioteca Matplotlib.
        """

        plt.figsize=(6, 4)
        self.df.hist()
        plt.show()
        
        colors = [self.inv_name_dict[item] for item in self.df['Species']]

        scatter = plt.scatter(self.df['SepalLengthCm'], self.df['SepalWidthCm'], c = colors)
        plt.xlabel('sepal length (cm)')
        plt.ylabel('sepal width (cm)')
        plt.legend(handles=scatter.legend_elements()[0],
        labels = self.inv_name_dict.keys())
        plt.show()

    def Treinamento(self):
        """
        Treina o modelo de machine learning.

        Detalhes:
            * Utilize a função `train_test_split` para dividir os dados em treinamento e teste.
            * Escolha o modelo de machine learning que queira usar. Lembrando que não precisa ser SMV e Regressão linear.
            * Experimente técnicas de validação cruzada (cross-validation) para melhorar a acurácia final.
        
        Nota: Esta função deve ser ajustada conforme o modelo escolhido.
        """
        X = self.df[['PetalLengthCm', 'PetalWidthCm']]
        y = self.df['Species'].map(self.inv_name_dict)

        X_train, self.X_test, y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        self.X_test_scaled = scaler.transform(self.X_test)



        svm = SVC()
        svm.fit(X_train_scaled, y_train)

        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(X_train_scaled, y_train)

        self.modelos = {'SVM': svm, 'KNN': knn}
        

    def Teste(self):
        """
        Avalia o desempenho do modelo treinado nos dados de teste.

        Esta função deve ser implementada para testar o modelo e calcular métricas de avaliação relevantes, 
        como acurácia, precisão, ou outras métricas apropriadas ao tipo de problema.
        """
        for nome_modelo, modelo in self.modelos.items():
            y_pred = modelo.predict(self.X_test_scaled)
            print(f"Resultados para {nome_modelo}:")
            print("Acurácia:", accuracy_score(self.y_test, y_pred))
            print("Precisão:", precision_score(self.y_test, y_pred, average='weighted'))
            print("Recall:", recall_score(self.y_test, y_pred, average='weighted'))
            print("F1-score:", f1_score(self.y_test, y_pred, average='weighted'))
            print("Matriz de confusão:\n", confusion_matrix(self.y_test, y_pred))
            # Visualizar a matriz de confusão
            ConfusionMatrixDisplay.from_estimator(modelo, self.X_test_scaled, self.y_test)
            plt.show()

    def Train(self):
        """
        Função principal para o fluxo de treinamento do modelo.

        Este método encapsula as etapas de carregamento de dados, pré-processamento e treinamento do modelo.
        Sua tarefa é garantir que os métodos `CarregarDataset`, `TratamentoDeDados` e `Treinamento` estejam implementados corretamente.
        
        Notas:
            * O dataset padrão é "iris.data", mas o caminho pode ser ajustado.
            * Caso esteja executando fora do Colab e enfrente problemas com o path, use a biblioteca `os` para gerenciar caminhos de arquivos.
        """
        self.CarregarDataset("iris.data")  # Carrega o dataset especificado.

        # Tratamento de dados opcional, pode ser comentado se não for necessário
        self.TratamentoDeDados()

        # Visualizações dos dados 
        self.Visualizacoes()

        # Executa o treinamento do modelo
        self.Treinamento() 

        # Teste realiza as ações de mensuração do modelo
        self.Teste()

# Lembre-se de instanciar as classes após definir suas funcionalidades
# Recomenda-se criar ao menos dois modelos (e.g., Regressão Linear e SVM) para comparar o desempenho.
# A biblioteca já importa LinearRegression e SVC, mas outras escolhas de modelo são permitidas.

# Instanciando e executando o modelo

modelo = Modelo()
modelo.Train()