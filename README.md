# Szöveg- és webbányászat beadandó

  

## Bevezetés

A feladathoz az interneten talált [Simpsons Családos datasetet](https://www.kaggle.com/pierremegret/dialogue-lines-of-the-simpsons) szerettem volna elemezni és a következő kérdésekre próbáltam választ kapni:

1. Milyen pontossággal mondható meg kizárólag a szövegből, hogy ki szájából hangzik el?

2. Van-e olyan karakter aki nagyon jellegzetesen beszél, így könnyű a szövegből kitalálni, hogy ő mondta?

  

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

### Osztályozás
A feladathoz 3 Classifiert választottam (MultinomialNB, KNeighborsClassifier, SGDClassifier).

Ezekhez készült 1-1 optimalizáló függvény, ami a paraméterként megadott pontossági mértékre optimalizál (precision vagy accuracy).  Ezt úgy éri el, hogy kipróbálja az osztály általam kiválasztott paramétereinek az összes kombinációját (adott intervallumokat tekintve) és az osztály olyan példányát adja vissza, ami a legnagyobb pontosságot érte el.

## Eredmények
### 1. kérdés elemzése:
Elsőnek a 10 legszerepeltetettebb karaktert választottam ki, így a következő eredményt kaptam:

| Osztályozó      | Pontosság |
| ----------- | ----------- |
| MultinomialNB      | 38%   |
| KNeighborsClassifier   | 16%   |
| SGDClassifier   | 36% |

Ahogy a táblázat is mutatja, alacsony pontossággal lehet csak megmondani, hogy ki mondta az adott sort. A problémát a confusion mátrix is jól mutatja:
![MNB](https://github.com/Rombeii/Szoveg-es-webbanyaszat/blob/main/images/MNB_conf.png)
![SGD](https://github.com/Rombeii/Szoveg-es-webbanyaszat/blob/main/images/SGD_conf.png)

Érdekes lehet viszont azt vizsgálni, hogy melyik szereplőhöz melyik szót csatolja leginkább: [link](https://github.com/Rombeii/Szoveg-es-webbanyaszat/blob/main/most_important_words.txt)

Ez alapján azt is mondhatnánk, hogy nem lehet biztosan megmondani, hogy ki szájából hangzik el a szöveg, viszont érdekes tényeket tár fel, ha kisebb csapatokban próbáljuk osztályozni őket:

| Vizsgált karakterek | Pontosság | Kapcsolat | Megjegyzés |
| ----------- | ----------- | ----------- | ----------- |
| Homer Simpson, Marge Simpson | 69,57%       | Házastársak | Nincs nagy különbség a sorok számában |
| Bart Simpson, Lisa Simpson | 65,85%       | Testvérek | Nincs nagy különbség a sorok számában |
| Chief Wiggum, C. Montgomery Burns | 78,68%       | - | Nincs nagy különbség a sorok számában |
| Moe Szyslak, Ned Flanders | 73,39%       | - | Nincs nagy különbség a sorok számában |
| Lisa Simpson, C. Montgomery Burns | 79.20%  | - | 3x annyi szöveg tartozik Lisa Simpson-hoz, de a conf matrix alapján még talán elfogadható |

Az eredményekből jól látszik, hogy ha van valamilyen kapcsolat a karakterek között, akkor a szókincsük is bizonyos mértékig megegyezik, ezért nehezebb őket megkülönböztetni. Ugyanez eljátszható, ha 2 helyett 4 karaktert hasonlítunk össze:

| Vizsgált karakterek | Pontosság | Kapcsolat |
| ----------- | ----------- | ----------- |
| Homer Simpson, Marge Simpson, Bart Simpson, Lisa Simpson | 48,63%       | Egy család |
| C. Montgomery Burns, Moe Szyslak, Seymour Skinner, Ned Flanders | 59,65%       | - |

### 2. kérdés elemzése:
Annak a megválaszolására, hogy ki beszél a legjellegzetesebben a precision értéket választottam.

A következő eredményeket kaptam:

![MNB](https://github.com/Rombeii/Szoveg-es-webbanyaszat/blob/main/images/MNB_prec.png)
![SGD](https://github.com/Rombeii/Szoveg-es-webbanyaszat/blob/main/images/SGD_prec.png)
