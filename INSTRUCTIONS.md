# Instructions

Este arquivo contém instruções para a organização do código do trabalho.

O grupo deverá definir um identificador (`DSNAME`) para cada conjunto de dados, a fim de facilitar a organização e identificação dos arquivos.
Por exemplo, caso o conjunto de dados seja de reconhecimento de atividades humanas, o `DSNAME` poderia ser "HAR". 
No exemplo disponibilizado utilizei o identificador "sneakers".

Para cada conjunto de dados/tarefa (identificado por `DSNAME`), o repositório deve conter:

* Um arquivo chamado `dataset_DSNAME.py, com a implementação do `Dataset` e do `DataModule`.
Veja exemplos em `dataset_sneakers.py.

* Um arquivo chamado `dataset_DSNAME-1.about.ipynb`, com informações sobre o conjunto de dados e o `DataModule`.
  As informações sobre o conjunto de dados (como imagens e estatísticas) adicionadas ao relatório devem ser calculadas neste *notebook*.
  Pense neste *notebook* como uma bula sobre o conjunto de dados e a tarefa, e inclua informações relevantes para usuários do conjunto de dados.

* Um ou mais arquivos chamados `dataset_DSNAME-2.supervised.ipynb`, com código para treinar e avaliar um ou mais modelos de aprendizado de máquina supervisionado, sem o uso de SSL.
  Os resultados obtidos com este(s) *notebook(s)* devem ser comparados com os resultados de modelos pré-treinados com SSL.

* Um ou mais arquivos chamados `dataset_DSNAME-2.pretrain.ipynb`, com código para realizar o pré-treino dos *backbones*.
Como resultado, o *notebook* deve gerar *checkpoints* na pasta `log/DSNAME/...`, para serem reutilizados no(s) *notebook(s)* de avaliação.
  > Nota: você pode criar subpastas dentro da pasta `log/DSNAME/` para organizar os *checkpoints*.

* Um ou mais arquivos chamados `dataset_DSNAME-3.eval.ipynb`, com código para avaliar os *backbones* produzidos pelo pré-treino.
  Este *notebook* pode realizar o treino e a avaliação de modelos *downstream*, por exemplo.
  Como resultado, o *notebook* pode gerar arquivos CSV ou TXT na pasta `log/DSNAME/...` para análise posterior.

* Um ou mais arquivos chamados `dataset_DSNAME-3.results.ipynb`, com código para agregar os resultados gerados pelos *notebooks* de avaliação e produzir gráficos, tabelas e outros elementos utilizados no relatório.
  Caso necessário, esses elementos podem ser salvos na pasta `log/DSNAME/...`.

* Um script `bash` ou `python`, com uma sequência de comandos para executar todos os *notebooks* na ordem correta e produzir os resultados apresentados no relatório.

* Um arquivo `README.md`, contendo informações sobre os integrantes do grupo, os responsáveis pelas diferentes partes do trabalho e instruções sobre como reproduzir os resultados.


