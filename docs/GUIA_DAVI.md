# Guia do Davi: treinar a ResNet-50 com os dados do HD

Este guia é para quem recebeu o HD e vai treinar a ResNet-50 no próprio PC, sem
ter acompanhado a preparação dos dados. Siga os passos na ordem. Cada bloco de
comando está pronto para copiar; onde houver `<...>`, troque pelo seu valor.
A referência técnica completa (cada campo da config, cada arquivo de saída)
está em `docs/TREINO_RSNA.md`.

Os comandos aparecem em duas versões quando o sistema muda: **Linux** e
**Windows (PowerShell)**. Rode sempre a partir da pasta do repositório.

---

## O que mudou e o que recomendamos

### O que mudou desde a versão anterior do guia

- **Métrica de seleção: F2 por mama** (`metrica_selecao: f2_melhor`), no lugar do F1.
  - **Motivo:** neste projeto, um falso negativo (um câncer que passa) é muito mais grave que um falso positivo (uma paciente chamada de volta sem câncer). No F2, a sensibilidade pesa 4 vezes a precisão.
  - A seleção decide o early stopping, o `melhor.pt` e a escolha entre as tentativas do ajuste.
  - Continuam sendo calculados e reportados: `pf1`, F1, F1.5, `roc_auc`, `pr_auc`, sensibilidade com especificidade de 90% (`sens_esp90`) e especificidade com sensibilidade de 90% (`esp_sens90`). Cada F-beta tem o seu próprio limiar.
- **Orçamento por passos do otimizador, não por épocas.**
  - **Motivo:** a proporção de amostragem muda o tamanho da época. Com 1:10, uma época tem 8.580 imagens; com 1:1, só 1.560, cerca de 5,5 vezes menos. Por épocas, uma tentativa em 1:1 treinaria muito menos e a comparação entre proporções não valeria.
  - A unidade é a **época de referência**: os passos de uma época em 1:10, 239 passos com batch efetivo 36.
  - O ajuste tem 12 épocas de referência (2.868 passos), com validação a cada 239 passos, em qualquer proporção. O warmup, o scheduler e o early stopping usam a mesma escala.
- **EMA dos pesos** (`ema: true` na seção do modelo, decaimento 0,999). Com ela ligada, a validação e o `melhor.pt` usam os pesos da EMA. Desligada por padrão; é testada na fase 3 do ajuste.
- **Treino final com até 30 épocas de referência** (eram 25 épocas), com o mesmo early stopping (5 validações sem melhora).
- **Pré-treino escolhido livremente na config:** `timm_nome` aceita qualquer peso do timm (ex.: `resnet50.a1_in1k` em vez de `resnet50.tv2_in1k`). O nome usado fica registrado no run.
- **Caminhos da máquina num arquivo seu, fora do git:** `configs/local.yaml`, criado a partir do `configs/local.exemplo.yaml` (passo 6). As configs versionadas (`etapa2.yaml`, `treino.yaml`) não têm mais caminhos: não é preciso editar nem desfazer nada nelas antes do commit.
- **Checkpoints e MLflow no SSD**, fora do git (`checkpoints_dir` e `tracking_uri` no `local.yaml`). No `runs/` ficam só os arquivos pequenos.

### O que vamos fazer no ConvNeXt-Tiny (Jhonatan)

Ajuste na representação `cinza`, em três fases, cada uma com o mesmo orçamento por tentativa:

1. **Learning rate:** 1e-4, 5e-5 e 2e-5, com proporção 1:10 e EMA desligada, 2 seeds cada (6 tentativas).
2. **Proporção:** com o melhor lr, 1:1 (com `pos_weight` 1) e 1:4, 1 seed cada. O 1:10 já vem da fase 1.
3. **EMA:** a melhor combinação até ali, com a EMA ligada, 1 seed.

Nas fases 2 e 3, se a diferença para a melhor opção for pequena, a segunda seed roda só nas opções empatadas antes de decidir. A escolha é pelo F2, e cada fase termina com a tabela de todas as métricas no registro de experimentos.

Depois, o **treino final**: a config escolhida, com 3 seeds (1, 2 e 3) em cada representação (`cinza` e `realce_3canais`), 6 treinos.

### O que recomendamos para você na ResNet-50

São **recomendações**: você pode ajustar os hiperparâmetros da seção `resnet50` como achar melhor, **desde que registre todas as tentativas** no `docs/registro_experimentos_resnet50.md`, inclusive as ruins. As regras comuns aos dois modelos (métrica de seleção, orçamento, teste hold-out) ficam fixas; ver o passo 12.

- **Siga o mesmo plano de três fases** (passo 9):
  1. **Learning rate, com o otimizador SGD com Nesterov** (`otimizador: sgd`, momento 0,9, `weight_decay` 1e-4, como já está na seção `resnet50` da config): **1e-2, 5e-3 e 2e-3**, em 1:10 e sem EMA, com **2 seeds cada** (0 e 1).
     - **Por que esses valores:** a receita clássica da ResNet (Goyal et al., 2017) usa 0,1 para batch 256, o que dá cerca de 0,014 para batch 36 por escala linear. Essa é a escala do treino do zero; no fine-tuning a partir do ImageNet, o melhor costuma ficar nela ou abaixo. Os três valores cobrem um fator 5 a partir do padrão da config (0,01), como no ConvNeXt.
     - **Essa faixa só vale para SGD.** Se trocar para `otimizador: adamw`, a escala é outra, cerca de 100 vezes menor: teste 3e-4, 1e-4 e 5e-5, com `weight_decay` 0,01 (a faixa usual de fine-tuning com AdamW, a mesma ordem do ConvNeXt).
  2. **Proporção:** com o melhor lr, 1:1 (com `pos_weight` 1) e 1:4, 1 seed cada.
  3. **EMA:** a melhor combinação, com `ema=true`, 1 seed.

  Nas fases 2 e 3, desempate com a segunda seed como no ConvNeXt.
- **Ajuste o `batch_real` e a `acumulacao` para a sua GPU** (passo 8), com o batch efetivo perto de 36. O orçamento por passos se adapta ao batch sozinho.
- **Registre tudo** no `docs/registro_experimentos_resnet50.md`: a máquina, cada tentativa com a tabela do `rsna_comparar --ajuste`, a config final e os treinos finais.
- Depois do ajuste, o treino final é o mesmo do ConvNeXt: 3 seeds em cada representação.

### Lembretes do que não pode mudar

- **O teste hold-out só é usado no fim**, uma única vez e em conjunto. Nada de rodar `rsna_avaliar_teste.py`, ler o split `test` ou olhar métricas de teste.
- **Splits e dados intactos:** nada muda em `splits/rsna_fold0/`, nas pastas `rsna_fold0_etapa1` e `rsna_fold0_realce`, nem na Etapa 1 e na Etapa 2.
- **Sem flip horizontal:** a parede torácica fica sempre à esquerda.

---

## 1. Visão geral

**O que já está pronto (não refaça):**
- **Divisão por paciente** (`splits/rsna_fold0/`):
  - 15% de teste hold-out e 5 folds; só o fold 0 é a validação;
  - o treino tem todas as 780 positivas mais um pool fixo de 22.000 negativas.
- **Etapa 1** (pasta `rsna_fold0_etapa1`):
  - os DICOMs viraram PNG de 8 bits, com a máscara da mama;
  - parede torácica sempre à esquerda, lado maior de 1536 px;
  - 39.803 imagens.
- **Etapa 2** (no código): canvas de 768x512, augmentation e normalização do ImageNet, aplicados na hora do carregamento.
- **Realce** (pasta `rsna_fold0_realce`): a representação em 3 canais já calculada, com R = anatomia, G = top-hat de 0,6 mm e B = CLAHE.
- **Código de treino, avaliação e comparação**, já testado na máquina do Jhonatan com o ConvNeXt-Tiny.

**O que você vai fazer:**
1. Conferir a máquina, instalar o ambiente e copiar os dados.
2. Rodar o smoke test e o teste de sanidade com a ResNet-50.
3. Ajustar os hiperparâmetros da ResNet-50 em três fases (lr, proporção, EMA), só na validação.
4. Treinar a configuração escolhida nas **duas representações** (`cinza` e `realce_3canais`), com **3 seeds cada**: 6 treinos.
5. Preencher o registro de experimentos e devolver os resultados pelo git.

**O que você NÃO deve fazer** (detalhes no passo 12):
- Rodar qualquer coisa no teste hold-out.
- Mudar splits, dados, Etapa 1 ou Etapa 2.
- Mexer na seção `convnext_tiny` da config.
- Ligar o flip horizontal.
- Trocar a métrica de seleção.
- Commitar checkpoints.

---

## 2. Requisitos da máquina

| Item | Mínimo | Como conferir |
|------|--------|---------------|
| GPU NVIDIA | 8 GB de VRAM (menos funciona com batch menor) | `nvidia-smi` |
| Driver NVIDIA | **580 ou mais novo** para CUDA 13.0 (ou 560+ com o índice CUDA 12.6, ver passo 4) | `nvidia-smi` (canto superior: "Driver Version") |
| Espaço em SSD | **~80 GB** para os dados (21 GB da Etapa 1 + 58 GB do realce) e **~15 GB** para checkpoints e MLflow | Linux: `df -h`; Windows: `Get-PSDrive -PSProvider FileSystem` |
| Python | **3.12** | `python3.12 --version` (Linux) ou `py -3.12 --version` (Windows) |
| git | qualquer versão recente | `git --version` |
| Internet | no primeiro treino, para baixar os pesos do ImageNet (~100 MB) | |

Os dados precisam estar num **SSD**. Num HD, o carregamento das imagens fica mais lento que a GPU e o treino se arrasta.

---

## 3. Código

**Primeira vez** (clonar):
```bash
git clone https://github.com/ProjetoIntegrador-4Periodo/ProjetoIntegrador4Periodo.git
cd ProjetoIntegrador4Periodo
git switch feat/rsna-treino
```

**Se você já tem o repositório:**
```bash
cd ProjetoIntegrador4Periodo
git fetch origin
git switch feat/rsna-treino
git pull
```

**Sua branch de resultados** (crie uma vez, a partir da `feat/rsna-treino`):
```bash
git switch -c resultados/resnet50-davi
```
Todo o seu trabalho (config da ResNet-50, `runs/` e registro) vai para essa branch.

---

## 4. Ambiente

**Linux:**
```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements-treino.txt
```

**Windows (PowerShell):**
```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements-treino.txt
```

Se o seu driver for **mais antigo que 580** e não der para atualizar, troque
`cu130` por `cu126` nos dois sistemas (precisa de driver 560 ou mais novo).

**Conferir:**
```bash
python -c "import torch, timm, mlflow; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0), timm.__version__, mlflow.__version__)"
```
Esperado: `2.14.0+cu130 13.0 True <sua GPU> 1.0.29 3.16.1`. Se aparecer `False`, veja o passo 13 (erro de CUDA ou driver).

**Testes do repositório** (uns 15 s, sem GPU e sem dados):
```bash
python -m pytest -q
```
Esperado: todos `passed`, nenhum `failed`.

---

## 5. Dados

As **duas pastas precisam ir juntas para o mesmo SSD**. O realce usa a máscara que está na pasta da Etapa 1, então uma não funciona sem a outra. No exemplo, o HD está em `/media/hd` (Linux) ou `E:` (Windows), e o destino no SSD é `/dados/rsna` ou `D:\rsna`.

**Linux:**
```bash
mkdir -p /dados/rsna
rsync -a --info=progress2 /media/hd/rsna_fold0_etapa1 /media/hd/rsna_fold0_realce /dados/rsna/
```

**Windows (PowerShell):**
```powershell
robocopy E:\rsna_fold0_etapa1 D:\rsna\rsna_fold0_etapa1 /E /MT:8 /R:2 /W:5
robocopy E:\rsna_fold0_realce D:\rsna\rsna_fold0_realce /E /MT:8 /R:2 /W:5
```
A cópia de ~80 GB leva de 15 minutos a 1 hora, dependendo do HD.

**Conferir a integridade** (cada pasta tem um `SHA256SUMS`):
```bash
python -m scripts.verificar_checksums /dados/rsna/rsna_fold0_etapa1
python -m scripts.verificar_checksums /dados/rsna/rsna_fold0_realce
```
Esperado:
- `OK: 79608 arquivos conferidos, todos íntegros.` na Etapa 1;
- `OK: 39804 arquivos conferidos, todos íntegros.` no realce.

Leva alguns minutos.

**Se falhar:** o script lista os arquivos que faltam ou que diferem.
1. Copie de novo só esses arquivos (ou a pasta inteira) e rode a verificação outra vez.
2. Se continuar falhando, o problema está no HD: avise o Jhonatan com a lista que o script mostrou.
3. **Não treine com uma pasta que não passou na verificação.**

---

## 6. Configuração

São dois arquivos: o `configs/local.yaml` (seu, fora do git) e o batch da ResNet-50 no `configs/treino.yaml`. Não mexa em mais nada.

**`configs/local.yaml`**: os caminhos e o nome da sua máquina. Crie a partir do exemplo versionado:

Linux:
```bash
cp configs/local.exemplo.yaml configs/local.yaml
```

Windows (PowerShell):
```powershell
Copy-Item configs/local.exemplo.yaml configs/local.yaml
```

Depois edite o `configs/local.yaml` com os seus caminhos:
```yaml
maquina: pc-davi

etapa2:
  etapa1_dir: /dados/rsna/rsna_fold0_etapa1          # Windows: D:/rsna/rsna_fold0_etapa1
  realce_dir: /dados/rsna/rsna_fold0_realce          # Windows: D:/rsna/rsna_fold0_realce
  verificacao_dir: /dados/rsna/etapa2_verificacao

treino:
  checkpoints_dir: /dados/rsna_treino/checkpoints     # Windows: D:/rsna_treino/checkpoints
  verificacao_dir: /dados/rsna_treino/verificacao

mlflow:
  tracking_uri: sqlite:////dados/rsna_treino/mlflow.db   # Windows: sqlite:///D:/rsna_treino/mlflow.db
  artefatos_dir: /dados/rsna_treino/mlartifacts
```
No Windows, use barras normais (`D:/...`) no YAML. A seção `pipeline_rsna` do exemplo é da divisão e da Etapa 1, que você não roda: pode deixar como está.

O `configs/local.yaml` está no `.gitignore`: ele nunca entra num commit. Se faltar um caminho, o script para com uma mensagem que aponta para o `local.exemplo.yaml`.

As pastas podem ficar em qualquer lugar do SSD; só os dois caminhos de dados
(`etapa1_dir` e `realce_dir`) importam. Os manifests e os `SHA256SUMS` guardam caminhos relativos à própria
pasta. O `config_usada.yaml` da Etapa 1 mostra os caminhos da máquina onde os
dados foram gerados (`/mnt/SSD2/...`): é só registro, nenhum código lê esses
caminhos, então não edite esse arquivo.

**`configs/treino.yaml`**: só o batch da ResNet-50 (e, no fim do ajuste, a config escolhida, passo 9). Não mude `modelo` nem `mlflow.experimento`: os comandos do guia já passam `--modelo resnet50`.
```yaml
modelos:
  resnet50:
    batch_real: 16     # o maior que coube no passo 8
    acumulacao: 2      # batch efetivo = 16 x 2 = 32 (perto de 36)
    # o resto, só no ajuste (passo 9)
```

**Regra do batch:** `batch_real` é o que cabe na sua GPU. Escolha a `acumulacao` para o batch efetivo (`batch_real x acumulacao`) ficar perto de 36. Exemplos:
- 12 x 3 = 36;
- 16 x 2 = 32;
- 8 x 4 = 32;
- 6 x 6 = 36.

### MLflow: banco SQLite e interface

O MLflow guarda os runs num **banco SQLite**, um arquivo só: o `mlflow.db`,
no caminho de `mlflow.tracking_uri`. Os artefatos (configs, predições e
curvas de cada run) ficam na pasta `mlflow.artefatos_dir`.

- **Não use uma pasta `mlruns/`.** O backend de pasta está desativado no MLflow 3.16.
- **Criação:** o banco e a pasta de artefatos são criados sozinhos no primeiro treino.
- **Formato da URI:**
  - Linux: `sqlite:////dados/rsna_treino/mlflow.db` (quatro barras: `sqlite:///` mais o caminho absoluto `/dados/...`);
  - Windows: `sqlite:///D:/rsna_treino/mlflow.db` (três barras e a letra do disco).

**Abrir a interface** (com o ambiente ativado, a partir de qualquer pasta):
```bash
# Linux
mlflow ui --backend-store-uri sqlite:////dados/rsna_treino/mlflow.db --port 5000
```
```powershell
# Windows
mlflow ui --backend-store-uri sqlite:///D:/rsna_treino/mlflow.db --port 5000
```
1. Abra `http://localhost:5000` no navegador e entre no experimento `rsna_fold0`.
2. Cada treino é um run, com params (hiperparâmetros e config da Etapa 2), métricas a cada validação (o eixo x é o passo do otimizador) e tags (`modelo`, `representacao`, `seed`, `modo`, `maquina`, `gpu`, `commit`, `checkpoint_dir`).
3. Feche com Ctrl+C no terminal.

Pode deixar a interface aberta durante o treino e atualizar a página para ver as validações novas.

**Cuidados:**
- **Mesmo endereço:** passe na `--backend-store-uri` exatamente o mesmo caminho do `tracking_uri` da config. Com outro caminho, o MLflow cria um banco novo e vazio, e parece que os runs sumiram.
- **Não apague nem copie o `mlflow.db` com um treino rodando.**
- **Sem MLflow:** com `mlflow.ativo: false`, os arquivos de `runs/` continuam sendo gravados, e a comparação usa só eles.

Se o seu ficar diferente de 36, tudo bem: ele é registrado nas saídas.

---

## 7. Smoke test e teste de sanidade

**Smoke test:** 1 época curta (500 imagens do treino e 200 da validação) nas duas representações. Ele mede a vazão, a VRAM e o tempo estimado por época.
```bash
python -m scripts.rsna_treino_verificar smoke --modelos resnet50
```
Deu certo se:
- termina sem erro e imprime uma tabela com duas linhas (`cinza` e `realce_3canais`);
- `loss` é um número (não `nan`);
- `vram_pico_gb` está abaixo da VRAM da sua GPU;
- `min_por_epoca_estimado` (uma época de referência, 8.580 imagens de treino + a validação) e `horas_30_epocas` dão a previsão do treino.

Na RTX 4060 do Jhonatan, a ResNet-50 deu cerca de 50 imagens/s no treino e 135 na validação, ou uns 4 min por época de referência (8.580 imagens de treino e 9.033 de validação).

**Teste de sanidade:** a ResNet-50 tem de conseguir decorar 64 imagens fixas (32 positivas). O teste roda sem augmentation, drop path nem weight decay, sem acumulação e com warmup curto (5 épocas, 30 passos). Ele só confirma que o laço de treino aprende; se não decorar, há algo errado no laço ou nos dados. Na RTX 4060, ela decorou entre 30 e 40 épocas (2 a 3 minutos), conforme a rodada: na GPU o treino não é determinístico.
```bash
python -m scripts.rsna_treino_verificar sanidade --modelos resnet50
```
Deu certo se:
- `decorou` é `True`;
- `loss_final` fica abaixo de 0,02;
- a lista `loss por época` cai de cerca de 0,7 para perto de 0.

Se `decorou` for `False`, pare e avise.

Os dois testes gravam só em `verificacao_dir`: nada vai para `runs/` nem para o MLflow.

---

## 8. Maior batch que cabe na GPU

```bash
python -m scripts.rsna_treino_verificar maior-batch --modelos resnet50
```
O script tenta `batch_real` 4, 8, 12, 16, 20, 24, 32... até faltar memória e imprime `resnet50: maior batch_real que cabe = N`.

**Use um degrau abaixo de N** se `vram_pico_gb` de N estiver a menos de 0,5 GB do total. O carregamento real usa um pouco mais de memória que o teste.

Na RTX 4060 (8 GB), coube 16 (5,5 GB) e 20 não coube. Grave o valor e a acumulação em `configs/treino.yaml` (passo 6).

---

## 9. Ajuste de hiperparâmetros

Cada tentativa é um treino curto avaliado só na validação, com o mesmo orçamento: **12 épocas de referência em passos** (2.868 passos com batch efetivo 36), com validação a cada época de referência e early stopping depois de 5 validações sem melhora. Na RTX 4060, isso dá cerca de 50 min por tentativa da ResNet-50.

A escolha é pela métrica de seleção, o **F2 por mama** (`f2_melhor`). Rode uma tentativa por comando; o número da tentativa é automático.

Os laços abaixo mostram os comandos. Para as horas de treino, **rode cada laço desacoplado do terminal**, como na subseção a seguir.

### Rodar desacoplado do terminal, acompanhar e retomar

Uma tentativa leva perto de 1 h e um laço inteiro, de 5 a 12 h. Se o laço rodar preso ao terminal (ou à sessão SSH, do VS Code ou de um assistente), **fechar a janela mata o treino**. Isso aconteceu na tentativa 1 do ConvNeXt: o processo caiu com a sessão no meio da noite e só voltou de manhã, pelo checkpoint. Por isso, rode os laços do ajuste e do treino final desacoplados, gravando um log.

**1. Ponha o laço num script**, numa pasta de logs fora do repositório (ex.: `/dados/rsna_treino/logs/`). Exemplo da fase 1:

Linux (`fase1.sh`):
```bash
#!/usr/bin/env bash
cd ~/ProjetoIntegrador4Periodo || exit 1
export PYTHONUNBUFFERED=1    # o log sai na hora, sem buffer
for lr in 1e-2 5e-3 2e-3; do
  for seed in 0 1; do
    echo "=== $(date '+%F %T') lr=$lr seed=$seed"
    .venv/bin/python -m scripts.rsna_treinar --modo ajuste --modelo resnet50 --representacao cinza \
      --seed $seed --set lr=$lr || echo "=== FALHOU lr=$lr seed=$seed (código $?)"
  done
done
echo "=== $(date '+%F %T') fim"
```

Windows (`fase1.ps1`):
```powershell
Set-Location C:\ProjetoIntegrador4Periodo
$env:PYTHONUNBUFFERED = "1"
foreach ($lr in "1e-2", "5e-3", "2e-3") {
  foreach ($seed in 0, 1) {
    "=== $(Get-Date -Format s) lr=$lr seed=$seed"
    .\.venv\Scripts\python.exe -m scripts.rsna_treinar --modo ajuste --modelo resnet50 --representacao cinza --seed $seed --set lr=$lr
    if ($LASTEXITCODE -ne 0) { "=== FALHOU lr=$lr seed=$seed (código $LASTEXITCODE)" }
  }
}
"=== $(Get-Date -Format s) fim"
```

O `|| echo FALHOU` (ou o `if`) faz o laço seguir para a próxima tentativa se uma falhar, deixando a falha marcada no log.

**2. Inicie desacoplado.** Escolha uma das opções:

- **Linux, `setsid nohup`** (sobrevive ao fechamento do terminal e ao fim da sessão SSH):
  ```bash
  cd /dados/rsna_treino/logs
  chmod +x fase1.sh
  setsid nohup ./fase1.sh >> fase1.log 2>&1 < /dev/null &
  ```
- **Linux, `tmux`** (dá para voltar à tela do treino depois):
  ```bash
  tmux new -s ajuste                          # abre a sessão
  ./fase1.sh 2>&1 | tee -a fase1.log          # roda lá dentro
  # Ctrl+B e depois D: sai da sessão, o treino continua
  tmux attach -t ajuste                       # volta para ela
  ```
- **Windows (PowerShell)**, processo em segundo plano, independente da janela:
  ```powershell
  cd D:\rsna_treino\logs
  Start-Process powershell -WindowStyle Hidden `
    -ArgumentList "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "D:\rsna_treino\logs\fase1.ps1" `
    -RedirectStandardOutput fase1.log -RedirectStandardError fase1_erros.log
  ```
  Pode fechar a janela do PowerShell. No Windows, também desligue a suspensão enquanto treina (`powercfg /change standby-timeout-ac 0`; volte depois com o valor anterior) e não faça logoff, que encerra os processos do usuário. Bloquear a tela não atrapalha.

Nenhuma das opções sobrevive a uma queda de luz ou a um reboot: para isso existe o `--retomar` (item 4).

**3. Acompanhe o log** (de qualquer terminal, quantas vezes quiser):

Linux:
```bash
tail -f /dados/rsna_treino/logs/fase1.log                      # Ctrl+C só para de acompanhar
grep -E "^===|validação|melhor validação|Traceback" /dados/rsna_treino/logs/fase1.log | tail -20
pgrep -af rsna_treinar                                         # ainda está rodando?
nvidia-smi                                                     # GPU em uso
```

Windows:
```powershell
Get-Content D:\rsna_treino\logs\fase1.log -Wait -Tail 20       # Ctrl+C só para de acompanhar
Select-String -Path D:\rsna_treino\logs\fase1.log -Pattern "^===|validação|Traceback" | Select-Object -Last 20
Get-Process python -ErrorAction SilentlyContinue               # ainda está rodando?
nvidia-smi
```

Cada tentativa escreve uma linha por validação (`validação N (passo P): ... f2_melhor ...`, com `*` quando melhora) e uma linha `melhor validação ...` no fim. O MLflow mostra as mesmas curvas (passo 6).

**Para interromper** de propósito: Linux, `pkill -f fase1.sh; pkill -f rsna_treinar`; Windows, `Stop-Process -Name python` (e feche o PowerShell oculto pelo Gerenciador de Tarefas). Depois, retome como no item 4.

**4. Se cair**, o `ultimo.pt` guarda a última validação concluída, e o treino continua dela no mesmo run do MLflow:

1. No log, ache a tentativa que caiu: a última linha `ajuste/resnet50_cinza_tentativa<N> | ...` sem `melhor validação` depois.
2. Retome essa tentativa com a **mesma seed e os mesmos `--set`**, mais `--retomar --tentativa <N>`:
   ```bash
   python -m scripts.rsna_treinar --modo ajuste --modelo resnet50 --representacao cinza --seed 0 --set lr=1e-2 --retomar --tentativa 1
   ```
   O log mostra `retomando de .../ultimo.pt: validação 7 de 12`.
3. Rode **só as tentativas que faltam**, sem `--retomar`. Edite o script para tirar as que já terminaram e ponha a retomada na primeira linha. **Não rode o laço inteiro de novo:** no ajuste, o número da tentativa é automático, e as tentativas concluídas seriam treinadas outra vez com números novos.

No treino final, o nome da pasta é fixo (`<modelo>_<rep>_seed<N>`), então o laço da seção 10 pode ser rodado de novo depois de uma queda: ele pula o que terminou e retoma o que caiu.

### Fases do ajuste

**Fase 1, learning rate** (otimizador SGD com Nesterov, o da config; 1:10, sem EMA, 2 seeds cada, 6 tentativas):

Linux:
```bash
for lr in 1e-2 5e-3 2e-3; do
  for seed in 0 1; do
    python -m scripts.rsna_treinar --modo ajuste --modelo resnet50 --representacao cinza --seed $seed --set lr=$lr
  done
done
```

Windows (PowerShell):
```powershell
foreach ($lr in "1e-2", "5e-3", "2e-3") {
  foreach ($seed in 0, 1) {
    python -m scripts.rsna_treinar --modo ajuste --modelo resnet50 --representacao cinza --seed $seed --set lr=$lr
  }
}
```

**Fase 2, proporção** (com o melhor lr da fase 1, 1 seed cada; o 1:10 já vem da fase 1):
```bash
python -m scripts.rsna_treinar --modo ajuste --modelo resnet50 --representacao cinza --seed 0 --set lr=<melhor> negativos_por_positivo=1 pos_weight=1
python -m scripts.rsna_treinar --modo ajuste --modelo resnet50 --representacao cinza --seed 0 --set lr=<melhor> negativos_por_positivo=4
```

**Fase 3, EMA** (a melhor combinação até ali, 1 seed):
```bash
python -m scripts.rsna_treinar --modo ajuste --modelo resnet50 --representacao cinza --seed 0 --set lr=<melhor> negativos_por_positivo=<melhor> ema=true
```

**Empates:** nas fases 2 e 3, se a diferença de F2 para a melhor opção for pequena (da ordem da diferença entre as duas seeds da fase 1), rode a seed 1 só nas opções empatadas, com os mesmos `--set`, antes de decidir.

**Tabela de cada fase:**
```bash
python -m scripts.rsna_comparar --ajuste
```
O script gera `runs/ajuste/ajuste.md` e `runs/ajuste/ajuste.csv`, com:
- uma linha por configuração, ordenada pelo F2 médio;
- a média entre seeds de todas as métricas e, entre parênteses, a diferença entre a maior e a menor seed;
- a melhor validação sobre as validações treinadas (ex.: `11/12`);
- a coluna "ainda melhorando": `sim` quando a melhor validação foi uma das 2 últimas do orçamento, sem early stopping. Um vencedor que ainda melhorava pode render mais no treino final, que vai até 30 épocas de referência.

Copie a tabela para o registro de experimentos ao fim de cada fase, com a decisão tomada.

**Onde ficam os resultados:**
- `runs/ajuste/resnet50_cinza_tentativa<N>/`: o número N é automático. A pasta tem `config_treino.yaml` (com o que você trocou em `alteracoes`), `historico.csv` e `metricas_finais.json`.
- **MLflow:** o run `ajuste/resnet50_cinza_tentativa<N>`, com o que foi trocado na tag `alteracoes`.

**No MLflow** (abra a interface como no passo 6, em "MLflow: banco SQLite e interface"):
1. Entre no experimento `rsna_fold0`.
2. Filtre com `tags.modo = "ajuste" and tags.modelo = "resnet50"`.
3. Marque as tentativas e clique em **Compare** para ver as curvas de `val_f2_melhor`, `val_roc_auc` e `loss_treino` lado a lado. O eixo x é o passo do otimizador, igual para todas as proporções.

**Outros hiperparâmetros:** se quiser ir além das três fases (por exemplo, `--set otimizador=adamw` com `lr` 3e-4, 1e-4 e 5e-5 e `weight_decay` 0,01, nunca a faixa do SGD; ou `peso_negativo_dificil`), fique à vontade. Use o mesmo orçamento e registre cada tentativa.

- **Representação:** ajuste com `cinza`. A config escolhida vale para as duas representações.
- **Como decidir:** pelo F2, olhando também `roc_auc`, `pr_auc` e `esp_sens90`. Diferenças menores que a diferença entre seeds são ruído; nesse caso, prefira a config mais simples.
- **Não mexa:** `metrica_selecao`, o orçamento (`orcamento`, `proporcao_referencia`, `ajuste.epocas`) e a seção `convnext_tiny`.

**Quando escolher:** grave a config final na seção `resnet50` do `configs/treino.yaml` (os valores do `--set` vencedor). O treino final não aceita `--set`, para o repositório guardar exatamente o que foi treinado.

**Se uma tentativa cair:** retome com `--retomar --tentativa <N>` e rode só as que faltam (item 4 de "Rodar desacoplado do terminal, acompanhar e retomar").

---

## 10. Treino final

São 6 treinos: as duas representações, com as seeds 1, 2 e 3. Cada um vai até 30 épocas de referência (7.170 passos com batch 36), com validação a cada época de referência e early stopping depois de 5 validações sem melhora.

**Linux:**
```bash
for rep in cinza realce_3canais; do
  for seed in 1 2 3; do
    python -m scripts.rsna_treinar --modo final --modelo resnet50 --representacao $rep --seed $seed
  done
done
```

**Windows (PowerShell):**
```powershell
foreach ($rep in "cinza", "realce_3canais") {
  foreach ($seed in 1, 2, 3) {
    python -m scripts.rsna_treinar --modo final --modelo resnet50 --representacao $rep --seed $seed
  }
}
```

**Rode desacoplado do terminal** (`setsid nohup`, `tmux` ou `Start-Process`, como na seção 9), com um laço que se retoma sozinho: ele pula os treinos concluídos, retoma com `--retomar` o que tiver `ultimo.pt` e começa os outros. Assim, depois de qualquer queda, basta iniciar o mesmo script de novo.

Linux (`final.sh`; troque `CKPT` pelo seu `checkpoints_dir`):
```bash
#!/usr/bin/env bash
cd ~/ProjetoIntegrador4Periodo || exit 1
export PYTHONUNBUFFERED=1
CKPT=/dados/rsna_treino/checkpoints
for rep in cinza realce_3canais; do
  for seed in 1 2 3; do
    nome=resnet50_${rep}_seed${seed}
    [ -f runs/$nome/metricas_finais.json ] && { echo "=== $nome já concluído"; continue; }
    extra=""; [ -f $CKPT/$nome/ultimo.pt ] && extra="--retomar"
    echo "=== $(date '+%F %T') $nome $extra"
    .venv/bin/python -m scripts.rsna_treinar --modo final --modelo resnet50 --representacao $rep --seed $seed $extra \
      || echo "=== FALHOU $nome (código $?)"
  done
done
echo "=== $(date '+%F %T') fim"
```

Windows (`final.ps1`):
```powershell
Set-Location C:\ProjetoIntegrador4Periodo
$env:PYTHONUNBUFFERED = "1"
$ckpt = "D:\rsna_treino\checkpoints"
foreach ($rep in "cinza", "realce_3canais") {
  foreach ($seed in 1, 2, 3) {
    $nome = "resnet50_${rep}_seed$seed"
    if (Test-Path "runs\$nome\metricas_finais.json") { "=== $nome já concluído"; continue }
    $extra = @(); if (Test-Path "$ckpt\$nome\ultimo.pt") { $extra = @("--retomar") }
    "=== $(Get-Date -Format s) $nome $extra"
    .\.venv\Scripts\python.exe -m scripts.rsna_treinar --modo final --modelo resnet50 --representacao $rep --seed $seed @extra
    if ($LASTEXITCODE -ne 0) { "=== FALHOU $nome (código $LASTEXITCODE)" }
  }
}
"=== $(Get-Date -Format s) fim"
```

Inicie com `setsid nohup ./final.sh >> final.log 2>&1 < /dev/null &` (Linux) ou com o `Start-Process` da seção 9 apontando para `final.ps1`, e acompanhe o `final.log` como lá.

**Tempo estimado:** na RTX 4060, cerca de 4 min por época de referência, ou seja, até 2 h por treino se ele for até as 30 (uns 12 h para os 6). O early stopping costuma parar antes. Use o `horas_30_epocas` do seu smoke test para a previsão na sua GPU. Rode um treino de cada vez (por exemplo, o laço inteiro à noite).

**Se cair** (queda de luz, reboot, sessão fechada, erro): inicie o `final.sh` (ou `final.ps1`) de novo. Para um treino avulso, rode o mesmo comando com `--retomar`. Ele continua da última validação concluída, no mesmo run do MLflow:
```bash
python -m scripts.rsna_treinar --modo final --modelo resnet50 --representacao realce_3canais --seed 2 --retomar
```
- **Treino já concluído:** se você rodar de novo um treino que já terminou, o script recusa e avisa. Nada é sobrescrito.
- **Laço interrompido:** o `final.sh` resolve sozinho. Sem ele, retome o treino interrompido com `--retomar` e depois rode os que faltam sem `--retomar`.

**Comparação** (depois dos 6 treinos):
```bash
python -m scripts.rsna_comparar
```
O script gera `runs/comparacao.md` e `runs/comparacao.csv`.

---

## 11. Devolução dos resultados

**Registro de experimentos:** preencha `docs/registro_experimentos_resnet50.md` **durante** o trabalho, não só no fim.
- **Seção 1 (máquina):** depois dos passos 7 e 8.
- **Seção 2 (tentativas):** uma linha a cada tentativa de ajuste concluída e, ao fim de cada fase, a tabela do `rsna_comparar --ajuste` e a decisão.
- **Seção 3 (config final):** quando escolher a config.
- **Seção 4 (treinos finais):** uma linha por treino final.
- **Seção 5 (observações):** sempre que algo fugir do roteiro (quedas, retomadas, erros).

Ele vai no mesmo commit dos resultados.

**O que commitar** (os arquivos pequenos de `runs/`; o `.gitignore` já deixa de fora o que não deve entrar):
```bash
git status                     # confira: só runs/, configs/treino.yaml e o registro
git add runs/ configs/treino.yaml docs/registro_experimentos_resnet50.md
git commit -m "resultados(resnet50): ajuste e treinos finais nas duas representações"
git push -u origin resultados/resnet50-davi
```

**Não commite:**
- os checkpoints (`*.pt`, ficam no `checkpoints_dir`);
- o `mlflow.db` e o `mlartifacts/`;
- as pastas de dados;
- o `configs/local.yaml` (o `.gitignore` já deixa de fora; confira que ele não aparece no `git status`).

**Antes do push**, confirme que nenhum arquivo passa de 5 MB, que é o limite do CI:
```bash
git diff --cached --stat
```

**Avise:** depois do push, mande ao Jhonatan:
- o nome da branch;
- o `runs/comparacao.md`;
- qualquer coisa anotada na seção 5 do registro.

Guarde os checkpoints: eles são necessários para a avaliação final no teste, que é feita uma única vez, no fim e em conjunto.

---

## 12. Regras que não podem ser quebradas

1. **Não use o teste hold-out.**
   - Não rode o `scripts/rsna_avaliar_teste.py`.
   - Não leia o split `test` em notebook.
   - Não olhe métricas de teste.

   Todas as escolhas (hiperparâmetros, validação, limiar) são feitas só na validação. O teste é usado uma única vez, no fim.
2. **Não altere os splits nem os dados:** nada em `splits/rsna_fold0/`, nas pastas `rsna_fold0_etapa1` e `rsna_fold0_realce`, nem nos arquivos da Etapa 1 e da Etapa 2. Se achar um problema nos dados, anote e avise.
3. **Não ative o flip horizontal.** A parede torácica fica sempre à esquerda, e o resto da Etapa 2 depende disso. Na augmentation, não mude nada.
4. **Não mude as regras comuns** sem combinar antes: a métrica de seleção (`metrica_selecao: f2_melhor`), o orçamento (`orcamento: passos`, `proporcao_referencia`, `max_epocas`, `ajuste.epocas`) e a `paciencia`. Com regras diferentes, a comparação entre os modelos deixa de valer. Os hiperparâmetros da seção `resnet50` são seus.
5. **Não mexa na seção `convnext_tiny`.** Ela é do outro modelo.

---

## 13. Problemas comuns

**Falta de memória na GPU (`CUDA out of memory`, `OutOfMemoryError`)**
1. Diminua o `batch_real` um degrau (por exemplo, 16 para 12) e aumente a `acumulacao` para manter o batch efetivo perto de 36.
2. Feche outros programas que usem a GPU (navegador com aceleração, jogos). Confira com `nvidia-smi`.
3. Rode `maior-batch` de novo.

**Carregamento lento (img/s bem abaixo do smoke test, GPU ociosa no `nvidia-smi`)**
- Confira se os dados estão no SSD, e não no HD.
- Ajuste `carregamento.workers` em `configs/etapa2.yaml`. Esse campo pode mudar; ele não altera os dados. Use de 4 a 8, em torno do número de núcleos físicos da CPU.
- No Windows, deixe o antivírus de fora da pasta dos dados (exclusão no Windows Defender): ele escaneia cada PNG lido.
- Se a RAM acabar (o sistema começa a usar swap), reduza os `workers`.

**Checksum divergente**
- Veja o passo 5: copie de novo os arquivos listados.
- Não treine com dados que não passaram na verificação.

**Erro de CUDA ou driver**
- `torch.cuda.is_available()` dá `False`, ou aparece `CUDA driver version is insufficient`: o driver é mais antigo que o CUDA do torch.
  - Atualize o driver NVIDIA (580 ou mais novo).
  - Ou reinstale o torch com o índice `cu126` (passo 4).
- `no kernel image is available`: a GPU não é suportada por essa versão do CUDA. Avise o Jhonatan.
- Erro ao baixar os pesos (`huggingface`, `timeout`): conexão com a internet. Rode de novo; depois do primeiro download, os pesos ficam em cache.

**MLflow**
- `database is locked`: duas execuções gravando ao mesmo tempo. Rode um treino por vez.
- Para o MLflow não atrapalhar, ponha `mlflow.ativo: false`. Os arquivos de `runs/` continuam sendo gravados normalmente.

**Treino recusa começar** (`ultimo.pt já existe` ou `já foi concluído`)
- Se você quer continuar, use `--retomar`.
- Se quer recomeçar do zero, apague a pasta do run em `runs/` e a do checkpoint, e anote o motivo no registro.

---

## 14. Checklist final

- [ ] `nvidia-smi`, Python 3.12 e git conferidos (passo 2)
- [ ] Repositório na `feat/rsna-treino` e branch `resultados/resnet50-davi` criada (passo 3)
- [ ] Ambiente instalado; `torch.cuda.is_available()` dá `True`; `pytest` todo verde (passo 4)
- [ ] As duas pastas copiadas para o mesmo SSD; os dois checksums deram OK (passo 5)
- [ ] `configs/local.yaml` criado a partir do exemplo, com os seus caminhos e `maquina`; batch gravado no `configs/treino.yaml` (passo 6)
- [ ] Smoke test sem erro nas duas representações (passo 7)
- [ ] Teste de sanidade com `decorou: True` (passo 7)
- [ ] Maior batch medido; `batch_real` e `acumulacao` gravados (passo 8)
- [ ] Laços do ajuste e do treino final rodados desacoplados do terminal, com log (`setsid nohup`, `tmux` ou `Start-Process`; seção 9)
- [ ] Fase 1 (lr, 2 seeds), fase 2 (proporção) e fase 3 (EMA) do ajuste feitas, com a tabela de cada fase no registro (passo 9)
- [ ] Config final gravada na seção `resnet50` e justificada no registro (passo 9)
- [ ] 6 treinos finais concluídos: `cinza` e `realce_3canais`, seeds 1, 2 e 3 (passo 10)
- [ ] `python -m scripts.rsna_comparar` rodado (passo 10)
- [ ] Registro de experimentos completo (passo 11)
- [ ] Commit só com `runs/`, `configs/treino.yaml` e o registro; sem checkpoints; push feito (passo 11)
- [ ] Jhonatan avisado, com a branch e o `comparacao.md` (passo 11)
- [ ] Teste hold-out não usado (passo 12)
