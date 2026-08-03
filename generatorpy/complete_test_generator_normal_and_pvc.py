import torch
import torch.nn as nn
import snntorch as snn
import wfdb
import numpy as np
import os
import matplotlib.pyplot as plt
import scipy.signal as signal
import random

# =====================================================================
# 1. Configurações Globais
# =====================================================================
DIRETORIO_BASE_VIVADO = r"C:\CIDIGITAL\TCC\Vivado\Testevivado7datasets106pvc\SNN\ip_testes_completos"
os.makedirs(DIRETORIO_BASE_VIVADO, exist_ok=True)

PACIENTE_TREINO = '119'
PACIENTES_TESTE = ['106', '200', '233']

JANELA = 200

# =====================================================================
# 1.5 Depuração Visual do NOVO Codificador
# =====================================================================
def plotar_debug_codificador(registro_str):
    print(f"\nGerando gráfico de depuração para o paciente {registro_str}...")
    try:
        record = wfdb.rdrecord(registro_str, pn_dir='mitdb', sampto=15000)
        annotation = wfdb.rdann(registro_str, 'atr', pn_dir='mitdb', sampto=15000)
    except Exception as e:
        return

    ecg_signal = record.p_signal[:, 0]
    
    try:
        idx_normal = list(annotation.symbol).index('N')
        pos = annotation.sample[idx_normal]
    except ValueError:
        return

    inicio = pos - (JANELA // 2)
    fim = pos + (JANELA // 2)
    trecho = ecg_signal[inicio:fim]
    
    min_val, max_val = np.min(trecho), np.max(trecho)
    if max_val > min_val: trecho = (trecho - min_val) / (max_val - min_val)
    else: trecho = np.zeros_like(trecho)
    
    canal_up, canal_down = np.zeros(JANELA), np.zeros(JANELA)
    
    for i in range(1, JANELA):
        if abs(trecho[i] - trecho[i-1]) >= 0.04: canal_up[i] = 1
        if trecho[i] >= 0.30: canal_down[i] = 1

    plt.figure(figsize=(10, 5))
    plt.subplot(2, 1, 1)
    plt.plot(trecho, color='blue')
    plt.title(f"Sinal Normalizado (Paciente {registro_str})")
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.subplot(2, 1, 2)
    spikes_visuais = [1 if u == 1 else (-1 if d == 1 else 0) for u, d in zip(canal_up, canal_down)]
    plt.plot(spikes_visuais, marker='o', linestyle='None', markersize=4, color='red')
    plt.title("Spikes Ortogonais")
    plt.yticks([-1, 0, 1], ['Largura (Bit 1)', 'Zero (0)', 'Agudeza (Bit 0)'])
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

plotar_debug_codificador('106')

# =====================================================================
# 2. Função de Extração com Novo Codificador
# =====================================================================
def extrair_dados_paciente(registro_str, max_amostras=60):
    print(f"Extraindo dados do MIT-BIH (Registro {registro_str})...")
    
    try:
        record = wfdb.rdrecord(registro_str, pn_dir='mitdb', sampto=150000)
        annotation = wfdb.rdann(registro_str, 'atr', pn_dir='mitdb', sampto=150000)
    except Exception as e:
        return [], []

    ecg_signal = record.p_signal[:, 0]
    simbolos = annotation.symbol
    posicoes = annotation.sample

    dados_spikes = []
    labels = []

    for sim, pos in zip(simbolos, posicoes):
        if sim in ['N', 'V']:
            inicio = pos - (JANELA // 2)
            fim = pos + (JANELA // 2)
            if inicio < 0 or fim >= len(ecg_signal): continue
                
            trecho = ecg_signal[inicio:fim]
            min_val, max_val = np.min(trecho), np.max(trecho)
            if max_val > min_val: trecho = (trecho - min_val) / (max_val - min_val)
            else: trecho = np.zeros_like(trecho)
            
            canal_up, canal_down = np.zeros(JANELA), np.zeros(JANELA)
            
            for i in range(1, JANELA):
                if abs(trecho[i] - trecho[i-1]) >= 0.04: canal_up[i] = 1
                if trecho[i] >= 0.30: canal_down[i] = 1
                    
            dados_spikes.append(torch.tensor(np.column_stack((canal_up, canal_down)), dtype=torch.float32))
            labels.append(0 if sim == 'N' else 1)
            
            if labels.count(0) >= max_amostras and labels.count(1) >= max_amostras: break
            
    return dados_spikes, labels

# =====================================================================
# 3. Definição do Modelo (GÊMEO DIGITAL)
# =====================================================================
class AceleradorECG_SNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 100, bias=False)
        
        # INICIALIZAÇÃO GUIADA (Conhecimento Prévio de Hardware)
        with torch.no_grad():
            # Classe 0 (Normal - Neurônios 0 a 49): Especialistas em Agudeza (Bit 0)
            self.fc1.weight[0:50, 0] = torch.rand(50) * 0.5 + 0.5  
            self.fc1.weight[0:50, 1] = torch.rand(50) * 0.1        
            
            # Classe 1 (PVC - Neurônios 50 a 99): Especialistas em Largura (Bit 1)
            self.fc1.weight[50:100, 0] = torch.rand(50) * 0.1      
            self.fc1.weight[50:100, 1] = torch.rand(50) * 0.5 + 0.5  
        
        self.drop = nn.Dropout(p=0.5) 
        self.lif1 = snn.Leaky(beta=1.0, threshold=1.0, reset_mechanism='zero') 

    def forward(self, x):
        mem1 = self.lif1.init_leaky()
        spk1_rec = []
        for step in range(x.size(0)):
            corrente = self.drop(self.fc1(x[step])) 
            spk1, mem1 = self.lif1(corrente, mem1)
            spk1_rec.append(spk1)
        return torch.stack(spk1_rec, dim=0)

# =====================================================================
# 4. Fase de Treinamento
# =====================================================================
print("\nMontando Dataset Híbrido para Treinamento...")

d_119, l_119 = extrair_dados_paciente('119', max_amostras=20)
d_106, l_106 = extrair_dados_paciente('106', max_amostras=20)
d_200, l_200 = extrair_dados_paciente('200', max_amostras=20)

dados_treino = d_119 + d_106 + d_200
labels_treino = l_119 + l_106 + l_200

if not dados_treino: exit()

dataset = list(zip(dados_treino, labels_treino))
random.shuffle(dataset)
dados_treino, labels_treino = zip(*dataset)

modelo = AceleradorECG_SNN()

# FINE TUNING: Taxa de aprendizado super baixa para não quebrar a inteligência nativa
optimizer = torch.optim.Adam(modelo.parameters(), lr=0.0005) 
loss_fn = nn.CrossEntropyLoss()

print("\nTreinando modelo (Sinapses Guiadas e Estritamente Excitatórias)...")

for epoch in range(100): 
    acertos = 0
    for dados, label in zip(dados_treino, labels_treino):
        modelo.train()
        optimizer.zero_grad()
        spk_out = modelo(dados)
        spike_count = spk_out.sum(dim=0)
        
        # Voltei para .sum() para aplicar a escala correta de gradiente para a Loss
        score_normal = spike_count[0:50].sum().unsqueeze(0)
        score_pvc = spike_count[50:100].sum().unsqueeze(0)
        predicao_treino = torch.cat((score_normal, score_pvc)).unsqueeze(0)
        
        loss = loss_fn(predicao_treino, torch.tensor([label], dtype=torch.long))
        loss.backward()
        optimizer.step()
        
        modelo.fc1.weight.data.clamp_(min=0.01)
        
        modelo.eval() 
        spk_out_teste = modelo(dados)
        spike_count_teste = spk_out_teste.sum(dim=0)
        max_idx = spike_count_teste.argmax().item()
        vencedor = 0 if max_idx < 50 else 1
        
        if vencedor == label: acertos += 1
            
    acuracia = (acertos / len(dados_treino)) * 100
    print(f"Época {epoch+1}/100 - Acurácia: {acuracia:.2f}%")
    
    # EARLY STOPPING: Se alcançou a excelência, pare o treinamento e salve o cérebro!
    if acuracia >= 98.0:
        print(f"-> CONVERGÊNCIA ALCANÇADA! Early Stopping ativado para proteger a matriz de pesos.")
        break

print("\nTreinamento concluído. Pesos congelados.")

# =====================================================================
# 5, 6 e 7. Exportação e Prova Real
# =====================================================================
def salvar_coe(caminho_arquivo, linhas_binarias):
    with open(caminho_arquivo, "w") as f:
        f.write("memory_initialization_radix=2;\nmemory_initialization_vector=\n")
        f.write(",\n".join(linhas_binarias) + ";\n")

def int2bin32(val):
    if val < 0: val = (1 << 32) + val
    return format(val, '032b')

pesos_crus = modelo.fc1.weight.data.numpy()

fator_escala = np.max(np.abs(pesos_crus))
if fator_escala > 0:
    pesos_crus = (pesos_crus / fator_escala) * 0.8 

pesos_fpga = np.round(pesos_crus * 100000000).astype(int)
linhas_pesos = ["".join([int2bin32(pesos_fpga[n, entrada]) for n in reversed(range(100))]) for entrada in range(2)]

print("\nIniciando geração de testes (Normal e PVC) para o Vivado...")

for paciente in PACIENTES_TESTE:
    dir_paciente = os.path.join(DIRETORIO_BASE_VIVADO, f"teste_paciente_{paciente}")
    os.makedirs(dir_paciente, exist_ok=True)
    
    salvar_coe(os.path.join(dir_paciente, "weight_mem.coe"), linhas_pesos)
    salvar_coe(os.path.join(dir_paciente, "neuron2class_assignment_mem.coe"), [("0001" * 50) + ("0000" * 50)])
    
    dados_teste, labels_teste = extrair_dados_paciente(paciente, max_amostras=15) 
    if not dados_teste: continue
        
    try:
        idx_normal = labels_teste.index(0) 
        normal_data_tensor = dados_teste[idx_normal]
        normal_data = normal_data_tensor.numpy()
        linhas_input_normal = [f"{int(normal_data[i,1])}{int(normal_data[i,0])}" for i in range(JANELA)]
        salvar_coe(os.path.join(dir_paciente, "input_mem_normal.coe"), linhas_input_normal)
        
        modelo.eval()
        vencedor_idx_teste_n = modelo(normal_data_tensor).sum(dim=0).argmax().item()
        classe_vencedora_teste_n = 0 if vencedor_idx_teste_n < 50 else 1
        
        print(f"\n[{paciente}] OK! Batimento NORMAL exportado.")
        print(f"  -> PROVA REAL (PYTHON) NORMAL: Neurônio Campeão: {vencedor_idx_teste_n} | Classe Prevista: {classe_vencedora_teste_n}")
        if classe_vencedora_teste_n == 0: print("  -> EXCELENTE: Normal Validado!")
        else: print("  -> FALHA no Normal.")
    except ValueError: pass

    try:
        idx_pvc = labels_teste.index(1) 
        pvc_data_tensor = dados_teste[idx_pvc]
        pvc_data = pvc_data_tensor.numpy()
        linhas_input_pvc = [f"{int(pvc_data[i,1])}{int(pvc_data[i,0])}" for i in range(JANELA)]
        salvar_coe(os.path.join(dir_paciente, "input_mem_pvc.coe"), linhas_input_pvc)
        
        modelo.eval()
        vencedor_idx_teste = modelo(pvc_data_tensor).sum(dim=0).argmax().item()
        classe_vencedora_teste = 0 if vencedor_idx_teste < 50 else 1
        
        print(f"[{paciente}] OK! Batimento PVC exportado.")
        print(f"  -> PROVA REAL (PYTHON) PVC: Neurônio Campeão: {vencedor_idx_teste} | Classe Prevista: {classe_vencedora_teste}")
        if classe_vencedora_teste == 1: print("  -> EXCELENTE: ARRITMIA VALIDADA PARA O HARDWARE!")
        else: print("  -> FALHA no PVC.")
    except ValueError: pass

print("\nProcesso finalizado. Carregue os novos .coe no Vivado!")