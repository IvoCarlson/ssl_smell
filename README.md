# ssl_smell

Esse trabalho é uma implementação que utiliza técnicas de Self-Supervised Learning como forma de melhorar os resultados obtidos com o SMELL para identificação de malwares.

# Como utilizar

## Preparação dos dados

A execução do código aqui presente depende da base de dados [MaleVis](https://web.cs.hacettepe.edu.tr/~selman/malevis/). Essa base se encontra disponível na Web portanto não foi adicionada a esse repositório. Uma vez obtida, o seu conteúdo deve ser extraído em um subdiretório 'db'.

Execute os módulos do notebook 'preparacao_arquivos_csv_treino_teste.ipynb' responsável por separar os dados em treino e teste.

Execute os módulos do notebook 'small_labeled_data.ipynb' responsável pela configuração dos rótulos para o modelo.

Execute os módulos do notebook 'EDA_MaleVis.ipynb' caso deseje uma visualização gráfica da base de dados e das transformações que lhe serão feitas.

## Treinamento dos modelos SSL

Aqui são detalhados os treinamentos do modelo SSL puro. Inicialmente é realizado um treinamento supervisionado simples. Em seguida, são treinadas as Proxy Tasks. Uma vez que elas foram treinadas, é realizada a transferência de conhecimento para um modelo supervisionado.

### Treinamento do modelo supervisionado

O treinamento do modelo supervisionado pode ser realizado com:

```bash
python3 train_sl.py
```

Pode ser necessário realizar alterações no arquivo de configuração do treinamento 'config_sl.yaml' disponível no subdiretório ssl/config.

### Treinamento das Proxy Tasks

Para o treinamento das Proxy Tasks foram gerados 3 arquivos. Um para a Proxy Task de Rotação, um para a Proxy Task de Posição Relativa e um para a Proxy Task de Ruído e Ofuscamento. Cada um desses tem seu próprio arquivo de configuração. A execução deles pode ser feita de forma similar ao treinamento do modelo supervisionado.

```bash
python3 train_ssl_rotation.py

python3 train_ssl_relpos.py

python3 train_ssl_noise.py

```

### Treinamento do modelo SL com pesos

Uma vez feito o treinamento das Proxy Tasks, os checkpoints salvos podem ser carregados para realizar um novo treinamento supervisionado. Também foi gerado um executável para essa tarefa para automatizar o processo de carregamento dos pesos.

```bash
python3 train_sl_proxy.py
```

### Resultados obtidos com cada Proxy Task
![alt text](https://github.com/IvoCarlson/ssl_smell/blob/main/images/SSL/RotationProxy.png)
![alt text](https://github.com/IvoCarlson/ssl_smell/blob/main/images/SSL/RelPosProxy.png)
![alt text](https://github.com/IvoCarlson/ssl_smell/blob/main/images/SSL/NoiseProxy.png)
![alt text](https://github.com/IvoCarlson/ssl_smell/blob/main/images/SSL/BlurProxy.png)

## Teinamento final

Execução do modelo SMELL

https://github.com/IvoCarlson/ssl_smell/blob/main/images/SMELL/Fig01.png
https://github.com/IvoCarlson/ssl_smell/blob/main/images/SMELL/Fig02.png

Execução do modelo SMELL com as representações obtidas pelo modelo auto-supervisionado

A implementação original dos autores de [Unsupervised Representation Learning by Predicting Image Rotations](https://arxiv.org/abs/1803.07728) pode ser encontrada [aqui](https://github.com/gidariss/FeatureLearningRotNet), também foi utilizada como referência a implementação de [Anuj Shah](https://github.com/anujshah1003) que pode ser encontrada [aqui](https://github.com/anujshah1003/self_supervised_learning_pytorch). A implementação original do autor de [Malware‐SMELL: A zero‐shot learning strategy for detecting zero‐day vulnerabilities](https://www.sciencedirect.com/science/article/abs/pii/S0167404822001808), por sua vez, pode ser encontrada [aqui](https://gitlab.com/sufex00/malware-smell)


<div align="center">

<a href="https://arxiv.org/abs/1803.07728"><img src="https://img.shields.io/badge/ArXiv-1803.07728-brightgreen"></a>
<a href="https://www.sciencedirect.com/science/article/abs/pii/S0167404822001808"><img src="https://img.shields.io/badge/Wiley-10.1002%2Fjcc.26027-brightgreen"></a>

</div>
