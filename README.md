# Sumarizacija sudskih presuda na srpskom jeziku primenom NLP metoda za ekstrakciju i apstrakciju teksta

## Tema projekta
- Sumarizacija sudskih presuda na srpskom jeziku pomoću NLP metoda
- Primena ekstraktivne i apstraktivne sumarizacije

## Implementacija
- Ekstraktivna sumarizacija primenom TextRank algoritma
- Apstraktivna sumarizacija primenom enkoder-dekoder arhitekture
    - Sequence-to-Sequence (2 LSTM)
    - Sequence-to-Sequence + Attention (BiLSTM + LSTM)
    - Teacher Forcing

## Skup podataka
- 400 sudskih presuda sa sažecima na srpskom jeziku preuzetih sa [sajta](https://e-case.eakademija.com/)
- Za evaluaciju rezultata korišćena je ROUGE metrika

## Pokretanje projekta
- Za izradu projekta korišćen je Python 3.10 i biblioteke navedene u fajlu requirements.txt
- Za trening Sequence2Sequence modela potrebno je pokrenuti python skriptu AbstractiveSummarizationSeq2Seq
- Za trening Sequence2Sequence + Attention modela pokrenuti python skriptu AbstractiveSummarizationSeq2SeqAttention
- Za pokretanje test modela pokrenuti python skriptu AbstractiveSummarizationSeq2SeqTestModel

## Rezultati
- Rezultati testiranja nad test i trening skupom mogu se naći u folderu [results](https://github.com/verak13/pravna-informatika-nn/tree/main/LegalCasesSummarization/results/s2s-25-vanilla)

## Članovi tima
- [Vera Kovačević R214/2021](https://github.com/verak13)
- [Nataša Ivanović R212/2021](https://github.com/natasa-ivanovic)

