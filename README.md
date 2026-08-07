# neuromorphic-neuron-biomedical
Implementação de um neurônio neuromórfico para processamento de sinais biomédicos

## Acelerador Neuromórfico para Processamento de Sinais de ECG em FPGA
Este repositório contém o código-fonte (SystemVerilog e Python) de um acelerador neuromórfico baseado em Redes Neurais Pulsantes (Spiking Neural Networks - SNNs) para a detecção de arritmias cardíacas em tempo real. O sistema foi projetado para operar com alta eficiência energética e processamento orientado a eventos, sendo validado fisicamente em uma FPGA Xilinx Artix-7.

As arquiteturas tradicionais de Inteligência Artificial demandam elevado poder computacional, o que inviabiliza sua aplicação em muitos dispositivos médicos embarcados. Este projeto propõe uma abordagem bioinspirada, substituindo operações matemáticas contínuas por pulsos discretos (*spikes*).

## Principais Características

* **Paridade Software-Hardware (Gêmeo Digital):** O modelo foi treinado em Python utilizando PyTorch e snnTorch. Os pesos sinápticos e limiares de ativação foram convertidos de ponto flutuante para aritmética inteira através de um fator de escala escalar, garantindo 100% de coerência de inferência no hardware.
* **Baixa Latência Determinística:** O processamento de uma janela de ECG completa (200 amostras) consome exatos 204 ciclos de clock (4 ciclos de inicialização de memória + 200 ciclos de integração), garantindo previsibilidade para aplicações de tempo real estrito.
* **Design Clínico Conservador (Fail-Safe):** A lógica de classificação foi arquitetada para priorizar a segurança médica. Em caso de empates na contagem de disparos neuronais, o módulo comparador favorece o alerta de anomalia cardíaca (Contração Ventricular Prematura - PVC), mitigando falsos negativos.
* **Máxima Eficiência de Silício:** A arquitetura evitou totalmente a utilização de multiplicadores em hardware, resolvendo o acúmulo sináptico com lógicas combinacionais otimizadas e consumindo 0 Blocos DSP na FPGA.

## Estrutura do Repositório
O projeto é dividido em dois ecossistemas principais:

### 1. Treinamento e Pré-Processamento (`/software`)
* **`gerador_de_testes_completos_normal_e_pvc.py`**: Script em Python responsável por extrair os sinais do banco MIT-BIH, aplicar o codificador Delta, treinar a rede LIF (com *leak* nulo) e exportar os parâmetros finais e matrizes de teste para arquivos `.coe`.

### 2. Acelerador Neuromórfico (`/hardware`)
Módulos descritos em SystemVerilog prontos para síntese no Vivado:
* **`snn.sv`**: Top Module do projeto, que instancia memórias, converte os estímulos temporais e orquestra a inferência.
* **`hidden_layer.sv`**: Controlador da camada oculta, que instancia os 100 neurônios de forma distribuída e paralela.
* **`neuron.sv`**: Unidade de processamento neural Integrate-and-Fire, responsável pela integração ponderada dos *spikes* sinápticos e controle do período refratário.
* **`max_comparer.sv`**: Árvore de comparadores otimizada encarregada de identificar o neurônio vencedor por taxa de disparo, incorporando a proteção *fail-safe* para empates.

## Síntese e Desempenho (Hardware Utilization)
Os resultados da síntese lógica para o dispositivo **Artix-7 (xc7a200tfbg484-3)** evidenciam o perfil de baixo consumo da arquitetura:

| Recurso da FPGA | Quantidade Utilizada | Utilização (%) |
| :--- | :--- | :--- |
| **LUTs** (Look-Up Tables) | 11.693 | 8,68% |
| **Flip-Flops** (Registradores) | 12.118 | 4,50% |
| **BRAM** (Block RAMs) | 51 | 13,97% |
| **DSPs** (Blocos Matemáticos) | 0 | 0,00% |
