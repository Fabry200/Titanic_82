# Predizione della Sopravvivenza del Titanic (82%)

Un classificatore K-Nearest Neighbors (KNN) su feature ridotte con PCA per predire la sopravvivenza dei passeggeri del Titanic con accuratezza > 80%.

---

## 📋 Indice

1. [Descrizione del Progetto](#descrizione-del-progetto)  
2. [Dataset](#dataset)  
3. [Ambiente e Installazione](#ambiente-e-installazione)  
4. [Utilizzo](#utilizzo)  
5. [Metodologia](#metodologia)  
6. [Risultati](#risultati)  
7. [Struttura dei File](#struttura-dei-file)  
8. [Contributi](#contributi)  
9. [Licenza](#licenza)  
10. [Contatti](#contatti)  

---

## 📝 Descrizione del Progetto

Questo progetto implementa una pipeline di machine learning per predire quali passeggeri del Titanic siano sopravvissuti. I passaggi chiave:

- **Caricamento e pulizia dei dati**  
- **Codifica delle feature** (`Sex`, `Pclass`, ecc.)  
- **Divisione train/test**  
- **PCA** per riduzione dimensionale (due componenti scelte)  
- **Implementazione custom di KNN**  
- **Ricerca iperparametri** su _k_  
- **Visualizzazione** dello spazio PCA e dei risultati  

Accuratezza test: **> 80%**.

---

## 📂 Dataset

- **Fonte:** `TitanicSurvival.csv`  
- **Colonne:**  
  - `nomi` — nome passeggero  
  - `sopravvissuto` — flag sopravvivenza (0 = no, 1 = sì)  
  - `sesso` — genere (male/female)  
  - `anni` — età in anni  
  - `classe_passe` — classe di viaggio (1, 2, 3)  

> **Nota:** Le righe con `anni` mancanti vengono scartate.

---

## 🛠 Ambiente e Installazione

1. Clona il repository:  
   ```bash
   git clone https://github.com/<tuo-username>/titanic-pca-knn.git
   cd titanic-pca-knn
