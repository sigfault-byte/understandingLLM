
# Results

```bash

Yomy, pitthical hyou vshon ble
ARD of cowngt we deel id enon ty orste. I seawat bapth his Hirmat gtsew r sinet dowth waftose
gexal, sousth,
ARsw.

AT:
Poto ly Rorrd weangh and and manl ppono id nteacrerd loth rt marar d dad myosrg.
Bt--HBevis fecfre
I sed,
Hr mish many hishou tand theliord oche ty sas fa s cis. RUT:
To as thaillgo
theandd dist ye fatr I shary ncik ndwev hadis og, stwnge h'dos sorno me lalled branlde
Rat lyof mal WARURCATourth thoror is mat sen de.

AMAUnd glme fournds.

LARD:
ou
```

With Karparty Tweaks:

```bash

S:
Powthon.

Yo; ore hren tire te or brorvos ovee th ry ar.

Hins thes oum' langorngu, hy wan'sh man nsosss me, to;
MEN:
SI esan to the htiavar.

An hok ingor ffldo onend freorlle gher.

LI hak:
Whar ak so'isout poru bor oukeneen, thamme'stt wat thand hull elay fou ty fo Sriket lyo shod the alkm southun I tlicy he!

An's toors het.

LO thigy;
Wem aris I who Shee tothavis ore jule;
Fons hig forwickes thyo erichee
Ferd. Heeiere ist theo goando fireak nge, or ng
Fr arevor fld mererese rrulld nse f
```

Multi head attention:

```bash

Sionowthon.

Lo; ore hrene dre.

ALLWINCOves ovee tu rous of you, thes bum'nl cequu velas wanw,
Ave casksss me,
At;
Mons 't es't to the hitai:
Cod she knid coulfldio, and freorlled ter.

LI hak: whar ands,
Sonet, bru bours kelayn, thatcerthipenter lof hull elay for thald Srik to he shou the alkm sonetur I thicy he ind prot, thoowt.

LOROLAD yo; mou, to de!, Sune,
That cerore trie;
Fons his frow Mous to foe ichee
Ferd. Houthee is no so goand my Moak pre, or ne
Fo my vor fort therese rrall freaze

```

Residual connection

```bash

Amman, lank,
feewhat srearmost O so fill and hoandent.

ROMENVORCUS:
Fring a racnot; it on yet.
-havl fied ning is goncic:
Uffe be tal be hadwill:
Es colfarwwearrs helad!

SOMy gold as to breed aff?
Preak I duke swill befound beebs, bran:
Mysarre-ba hing Lords.
You hath and continigh whis;
Awn a were homp,
Porther to the strest:
I that it in my nostely bood feirs the to tare well and share sion the wo frigh soud dukiss effes:
Gord.

HENENO:
Nand pourvert bee me-it deray
Jill to he broueds abl, p
```
---
After moving to :
    - B = 64
    - T = 256
    - C = 384
    - head = 6
    - layer = 6
    - max\_iters = 5000
    - lr = 3e-4
    
```bash
, borne more. Come on, Catesby, good made her:
And then Hereford, it die for me yourself
Before I crutched extempt me turning:
How far men of his wife were a thousand fled,
When I see, rancour here, we have lefter many:
This would I know write this power with his servery in-root
Mark. I heard the prite of your actor call,
Yet Cleotmis-graced Clifford shinely in time.

LADY CAPULET:
O, Edward:
God put up your hand: may do marry at a hangman your grace,
Shall call be conferm'd, as made a sock agai

```
This is **EXTRAORDINAIRE**

The model speaks non-sense but it is structure, there is intent and logic.




---
|                     | Entropy            | Cross -Entropy |
| ------------------- | ------------------ | -------------- |
| Bigram              | 4.894372904441695  | 2.4953         |
| Bigram SA(no tweak) | 4.6235283406864065 | 2.4392         |
| Bigram SA (tweak)   | 4.515100429786052  | 2.3995         |
| Bigram MSA          | 4.594609832702393  | 2.2683         |
| Bigram MSA + RC     | 4.711498269264631  | 2.0869         |
| Bigram Imba Karpathy tweaks |4.611705332367363   |1.4950  |


