# Instructions

Este arquivo contém instruções para a organização do trabalho.  

O grupo deverá definir um identificador (`DSNAME`) para cada conjunto de dados para facilitar a organização e identificação dos arquivos.
Por exemplo, caso o conjunto de dados seja de reconhecimento de atividades humanas, o `DSNAME` poderia ser "HAR". Utilizarei o identificador `"sneakers"` em meus exemplos.

Para cada conjunto de dados/tarefa (identificado por `DSNAME`), o repositório deve conter:

* um arquivo chamado `dataset_DSNAME.py` com a implementação do Dataset e do DataModule. Veja exemplos em `dataset_sneakers.py`.

* um arquivo chamado `dataset_DSNAME-1.about.ipynb` com informações sobre o conjunto de dados e o datamodule.
  As informações (p.ex., imagens) e estatísticas adicionadas no relatório devem ser calculadas neste *notebook*.
  Pense neste *notebook* como uma bula sobre o conjunto de dados e a tarefa e adicione informações relevantes para usuários do conjunto de dados.

* um ou mais arquivos chamados `dataset_DSNAME-2.supervised.ipynb` com código para treinar e avaliar um ou mais modelos de aprendizado de máquina de forma supervisionada sem o auxílio de SSL. 
  Os resultados obtidos com este(s) *notebook(s)* devem ser comparados com os resultados obtidos com modelos pré-treinados com SSL.

* um ou mais arquivos chamados `dataset_DSNAME-2.pretrain.ipynb` com código para realizar o pré-treino dos *backbones*. 
  Como resultado, o *notebook* deve produzir checkpoints na pasta `log/DSNAME/...` para que possam ser reusados no(s) *notebook(s)* de avaliação.
  > Nota: você pode criar sub-pastas dentro da pasta `log/DSNAME/` para organizar os checkpoints.

* um ou mais arquivos chamados `dataset_DSNAME-3.eval.ipynb` com código para avaliar os *backbones* produzidos pelo pré-treino. Este *notebook* pode realizar o treino e avaliação de modelos *downstream*, por exemplo.
  Como resultado, o *notebook* pode produzir arquivos CSV ou TXT na pasta `log/DSNAME/...` para análise posterior.

* um ou mais arquivos chamados `dataset_DSNAME-3.results.ipynb` com código para agregar os resultados produzidos pelos * notebooks*  de avaliação e gerar gráficos, tabelas, e outros elementos utilizados no relatório. 
  Caso necessário, os gráficos, tabelas e outros elementos podem ser salvos na pasta `log/DSNAME/...`.

* um *script* `bash` ou `python` com uma sequência de comandos para executar todos os *notebooks* na ordem correta e produzir os resultados apresentados no relatório.

* um arquivo `README.md` contendo informações sobre os responsáveis pelas diversas partes do trabalho e instruções de como reproduzir os resultados.



