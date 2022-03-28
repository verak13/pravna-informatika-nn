# Sumarizacija sudskih presuda na srpskom jeziku primenom NLP metoda za ekstrakciju i apstrakciju teksta
 Projekat iz predmeta pravna informatika
 
Tema projekta je sumarizacija sudskih presuda na srpskom jeziku pomoću NLP metoda. Kako su sudske presude veoma obimni dokumenti, primenjena je ekstraktivna sumarizacija (izdvajanje najbitnijih rečenica), a zatim i apstraktivna (generisanje novog teksta na osnovu psotojećeg). Ekstraktivna sumarizacija odrađena je primenom *Text Rank* algoritma, a za apstraktivnu sumarizaciju predložena su dva modela – *Sequence2Sequence* model i prošireni *Sequence2Sequence* model sa *Attention* mehanizmom – u ove arhitekture uključeni su *LSTM*,  *Bidirectional LSTM* i *Teacher forcing* mehanizam. *Sequence2Sequence* model čine dve rekurentne neuronske mreže, u ovom slučaju dve *LSTM* mreže - endkoder i dekoder. Enkoder procesira svaki element iz ulazne sekvence i generiše reprezentaciju ulaza koja se dalje prosleđuje dekoderu, čija uloga je da generiše izlaznu sekvencu. Arhitektura je prikazana na slici.

![image](https://user-images.githubusercontent.com/50635161/160447331-e1dc22ce-0ede-4a40-8396-509d897c0dbd.png)

Skup podataka koji je pripremljen za treniranje sastoji se od 400 sudskih presuda na srpskom jeziku (od kojih je 20 izdvojeno u test skup) i njihovih referentnih sižea i dostupan je na sajtu *https://e-case.eakademija.com/*. Zbog hardverskih ograničenja, modeli su trenirani na podskupu od 25 sudskih presuda. Za evaluaciju rezultata korišćena je *ROUGE* metrika.

Za izradu projekta korišćen je *Python 3.10* i biblioteke navedene u fajlu *requirements.txt*.
 
Za pokretanje treniranja *Sequence2Sequence* modela i *Sequence2Sequence* modela sa *Attention* mehanizmom potrebno je pokrenuti *python* skripte *AbstractiveSummarizationSeq2Seq* i *AbstractiveSummarizationSeq2SeqAttention*, a za pokretanje test modela pokrenuti skriptu *AbstractiveSummarizationSeq2SeqTestModel*.

Rezultati testiranja nad test i trening skupom mogu se naći u folderu [*results*](https://github.com/verak13/pravna-informatika-nn/tree/main/LegalCasesSummarization/results/s2s-25-vanilla).
 
Članovi tima:
- Nataša Ivanović R212/2021
- Vera Kovačević R214/2021
