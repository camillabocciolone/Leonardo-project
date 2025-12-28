
# Leonardo-project — EEGPT vs FDA (LL2 EEG)

Questo repository contiene notebook Colab per confrontare un **foundation model EEG (EEGPT)** con una **baseline FDA / PCA** e con **baseline spettrali (bandpower)** sul dataset LL2 (EEG workload).

## Contenuto

- **`eegpt_vs_FDA.ipynb`**  
  Notebook principale: pipeline completa per
  - caricamento + preprocessing + windowing
  - split subject-independent (recording-level / LOSO)
  - estrazione embedding EEGPT
  - feature FDA (PCA per-canale e PCA globale)
  - classificatore downstream (MLP) + confusion matrix
  - visualizzazioni (t-SNE/[UMAP])

- **`ll2egpt.ipynb`**  
  Notebook di esperimenti/miglioramenti per EEGPT:
  - tuning parametri patch/stride
  - test su robust z-score (MAD) per epoca
  - separazione task (Arithmetic vs Stroop)
  - controlli su preprocessing / timestamp / interpolazione
  - analisi diagnostiche
 
- **`baselinebandpower.ipynb`**  
  Scopo principale:
  - usare baseline bandpower / relative bandpower per capire quanti classi considerare e se il preprocessing è adeguato
  4 contesti:
    1) preprocessing senza fare interpolazione e 4 classi
    2) preprocessing con interpolazione e 4 classi
    3) preprocessing senza fare interpolazione e 3 classi
    4) preprocessing con interpolazione e 3 classi
  - risulta che la combinazione migliore sia preprocessing senza fare interpolazione e 3 classi
  - sicuramente si considerassero 2 classi sarebbe ancore migliore la predizione

---

## Setup rapido (Colab)

### 1) Monta Google Drive
Nel notebook è previsto il mount:
```python
from google.colab import drive
drive.mount('/content/drive')
````

### 2) Dataset (path atteso)

I notebook assumono che i file siano disponibili in una cartella tipo:

```
/content/drive/MyDrive/LL2/LL2/raw_data/
```

Sono supportati i due layout:

* `{level}_Data/{task}-{subject}.txt`
* `{task}_Data/{level}-{subject}.txt`

Esempi:

* `Stroop_Data/midlevel-7.txt`
* `midlevel_Data/Stroop-7.txt`

> Se il tuo path è diverso, modifica la variabile `data_dir` nel notebook.

### 3) Checkpoint EEGPT

Il checkpoint viene letto da Drive, ad esempio:

```
/content/drive/MyDrive/EEGPT/checkpoint/eegpt_mcae_58chs_4s_large4E.ckpt
```

---

## Pipeline (in breve)

1. **Preprocessing**

   * bandpass (default 0.5–45 Hz)
   * notch 50 Hz
   * detrend
   * clip robusto (quantile)
   * resampling 250 → 256 Hz
   * epoching 4s con stride 2s

2. **Split**

   * baseline: split recording-level con vincolo subject-independent
   * TODO/next: LOSO puro (Leave-One-Subject-Out) su tutti i soggetti

3. **Feature extraction**

   * **EEGPT**: `forward_features` → embedding per finestra → pooling per recording
   * **FDA/PCA**:

     * per-canale: PCA su 1024 campioni per canale, K PC per canale → 8K feature
     * globale: PCA su vettore (8×1024) → K feature
   * **Bandpower baseline**:

     * log bandpower in delta/theta/alpha/beta per canale
     * opzionale: relative bandpower

4. **Downstream classifier**

   * MLP (stesso classificatore per confronti fair)
   * confusion matrix + classification report
   * visualizzazioni: t-SNE / UMAP (record-level o window-level)

---

## Note importanti (per evitare leakage)

* La valutazione deve rimanere **subject-independent**:

  * tutte le finestre dello stesso soggetto/recording devono stare nello stesso split
* Le trasformazioni tipo PCA/standardizzazione vanno **fit solo sul train** e applicate a val/test.

---

## Obiettivo del progetto

Valutare se, in regime **low-data** (pochi soggetti), un foundation model EEG fornisce vantaggi rispetto a:

* baseline classiche e interpretabili (FDA/PCA)
* baseline spettrali (bandpower)

e usare FDA anche come supporto interpretativo (feature comprensibili / correlazioni con output).

---

## Dipendenze principali

I notebook installano automaticamente (in Colab):

* `numpy`, `scipy`, `scikit-learn`, `torch`
* `einops`, `tqdm`
* (opzionale) `umap-learn` per UMAP



---

## Contatti

Camilla Bocciolone



