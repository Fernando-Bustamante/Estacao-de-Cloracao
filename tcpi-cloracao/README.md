# Controle de uma estação de cloração

Simulação em Python do trabalho prático de Técnicas de Controle de Processos Industriais (ELT130, UFMG).

O processo tem dois reservatórios (água e hipoclorito) que alimentam dois tanques de mistura em tronco de cone. Em cada tanque são controlados o nível e a concentração de cloro. Foram implementadas duas estruturas: um MPC adaptativo e uma estrutura de contingência com PIs, controle de razão e override.

## Arquivos

- `simulador.py`: modelo não linear da planta, analisador de cloro e integração (RK4)
- `controladores.py`: PIs, sintonias, controle de razão, override e a estrutura de contingência
- `mpc.py`: MPC adaptativo com filtro de Kalman
- `parte1_validacao.py` a `parte7_cenarios.py`: scripts que reproduzem os resultados de cada seção do relatório

## Como rodar

```
pip install -r requirements.txt
python parte7_cenarios.py
```

Cada script imprime os números do relatório no terminal. A `parte2` e a `parte7` também salvam as figuras em PNG.

## Autores

Felipe Henrique Mendonça Carvalho, Fernando de Souza Bustamante e Rodrigo Ferreira Notini
