# Szöveg- és webbányászat beadandó

  

## Bevezetés

A feladathoz az interneten talált [Simpsons Családos datasetet](https://www.kaggle.com/pierremegret/dialogue-lines-of-the-simpsons) szerettem volna elemezni és a következő kérdésekre próbáltam választ kapni:

- Milyen pontossággal mondható meg kizárólag a szövegből, hogy ki szájából hangzik el?

- Van-e olyan karakter aki nagyon jellegzetesen beszél, így könnyű a szövegből kitalálni, hogy ő mondta?

  

## Előkészítés

A feladathoz az eddig ismert package-k mellett szükség lehet még az imblearn package-re is. Ha az anaconda navigator-ból indítjuk a spyder-t akkor érdemes lehet a conda-val telepíteni:

```

conda install -c conda-forge imbalanced-learn

```

  

## Megvalósítás

### Előfeldolgozás
Problémák:
- A dataset 6272 db egyedi szereplőt tartalmaz, viszont ennek egy jó részéhet csak kevés szöveg köthető.
- Sok olyan szöveg van ami csak 1-2 szóból áll.

Megoldások:
- Használjuk csak a *{NUM_OF_CHARACTERS}* legtöbbet megszólalót!
- Használjuk csak azokat a sorokat amik *{MIN_WORD_COUNT}* szónál hosszabbak!

### Implementáció
#### Vektorizálás
A megoldáshoz a CountVectorizer -t használtam (jobb eredményt adott, mint a TfidfVectorizer).
### Oversampling
Ha elég sok szereplőt használunk az osztályozáshoz, akkor akár nagy különbségek is kialakulhatnak a  szövegszámokban, ami problémát okozhat az osztályozás során. Ebben segít az imblearn package SMOTE és RandomOverSampler oversampler-je.

A feladathoz 3 Classifiert választottam (MultinomialNB, KNeighborsClassifier, SGDClassifier).

Ezekhez készült 1-1 optimalizáló függvény, ami a paraméterként megadott pontossági mértékre optimalizál (precision vagy accuracy).  Ezt úgy éri el, hogy kipróbálja az osztály általam kiválasztott paramétereinek az összes kombinációját (adott intervallumokat tekintve) és az osztály olyan példányát adja vissza, ami a legnagyobb pontosságot érte el.

Ezek az optimalizáicók sok időbe kerülnek, így összegyűjtöttem egy txt-be, hogy milyen konfigurációval érte el 1-1 Classifier a legjobb értéket.

