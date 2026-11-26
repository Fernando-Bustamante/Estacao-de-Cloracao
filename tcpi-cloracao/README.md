# Controle de uma estação de cloração

Simulação em Python do trabalho prático de Técnicas de Controle de Processos Industriais (ELT130, UFMG).

O processo tem dois reservatórios (água e hipoclorito) que alimentam dois tanques de mistura em tronco de cone. Em cada tanque são controlados o nível e a concentração de cloro. Foram implementadas duas estruturas: um MPC adaptativo e uma estrutura de contingência com PIs, controle de razão e override.

## Como rodar

```
git clone https://github.com/Fernando-Bustamante/Estacao-de-Cloracao
cd Estacao-de-Cloracao\tcpi-cloracao
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python sinotico.py
```

Se o `Activate.ps1` der erro de permissão, rode uma vez `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` e tente de novo.

## Usando a tela

A tela mostra o processo em funcionamento, com gráficos de nível, cloro e comandos. Pelos painéis de baixo é possível:

- trocar entre MPC e contingência com a planta rodando;
- mudar as referências de cloro, nível e produção;
- aplicar perturbações (consumo de cloro, concentração do hipoclorito, pressão da rede);
- simular a falha da válvula de descarga travada;
- simular a falha do otimizador do MPC ou do analisador de cloro do tanque C ("Otimizador C falha", "Analisador C sem sinal"): com o MPC ligado, a estação passa sozinha à contingência, sem salto nos comandos, e o seletor do topo muda para "Contingência (PI)";
- usar os cenários prontos ("Rede pressurizada", "Cloro +0,2 mg/L", "Tudo nominal").

O MPC também troca sozinho para a contingência quando o otimizador deixa de convergir de verdade, como acontece no cenário "Rede pressurizada".

O quadro "O que está acontecendo" explica cada decisão do controle, e passar o mouse sobre um equipamento mostra a função dele. As perturbações agem só no tanque C, e o tanque D serve de comparação.

## Arquivos

- `sinotico.py`: tela interativa
- `simulador.py`: modelo não linear da planta e do analisador de cloro
- `controladores.py`: PIs, controle de razão, override e a estrutura de contingência
- `mpc.py`: MPC adaptativo com filtro de Kalman e troca automática para a contingência
- `parte1_validacao.py` a `parte8_troca.py`: scripts que reproduzem os números e as figuras do relatório

## Autores

Felipe Henrique Mendonça Carvalho, Fernando de Souza Bustamante e Rodrigo Ferreira Notini
