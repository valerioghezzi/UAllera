# Affidabilità, evidenze e prestazioni

Stato della versione di sviluppo `0.3.0-dev.4`, 20 ottobre 2026. Questa guida descrive i controlli implementati, le prove eseguite e i limiti che il tutor deve considerare. I risultati del modello linguistico e i test del software restano verifiche distinte.

## Utilizzare il controllo

Nel pacchetto Windows della propria architettura, estrarre tutti i file e aprire `Avvia_UAllera.cmd`. Il runtime Python è incluso. Se si utilizza Minerva, preparare prima gli asset pubblici con la funzione dedicata: la prima preparazione richiede la rete; l'analisi successiva utilizza i file locali verificati. Il controllo deterministico e il benchmark descritti qui non richiedono un modello.

Selezionare lo ZIP con le cartelle degli studenti. Il rapporto HTML comprende ogni consegna individuata, le associazioni OMV–Word, le ambiguità e le parti non verificabili. Gli archivi annidati sono segnalati senza espansione automatica. Le copie originali restano intatte.

Dal sorgente, oppure con il runtime del pacchetto al posto di `python`, è possibile produrre rapporto e diagnostica locale:

```bash
python audit_cli.py consegne.zip -o Rapporto_esercitazioni.html --forensic Diagnostica.json
```

L'opzione `--forensic` crea un JSON distinto da sorgente e rapporto. L'HTML mostra riepilogo, rilievi e citazioni; il JSON conserva il catalogo strutturato completo delle evidenze. Per confrontare il comportamento con e senza riutilizzo del parsing:

```bash
python audit_cli.py consegne.zip -o Rapporto_senza_cache.html --forensic Diagnostica_senza_cache.json --no-cache
```

La CLI esegue solo i controlli deterministici quando non si specifica un motore locale. La configurazione di un motore, il completamento di una richiesta e l'effettiva qualità della risposta sono tre condizioni da controllare separatamente.

## Griglia di valutazione per il tutor

| Esito o segnale | Che cosa consente di concludere | Verifica del tutor |
|---|---|---|
| Verificato | Il controllo specifico ha trovato le evidenze richieste entro il proprio perimetro. | Controllare requisito, risultato atteso e posizione; non estendere l'esito all'intera relazione. |
| Mancante | Il requisito controllabile non risulta presente nel contenuto leggibile esaminato. | Distinguere output mancante, nascosto, vuoto o non salvato; esaminare la copertura. |
| Errato | Un contenuto osservato contraddice il requisito o il riferimento utilizzato dal controllo. | Riscontrare dati, gruppi, arrotondamenti e condizioni della consegna. |
| Da verificare | La decisione richiede un controllo umano o la risoluzione di un'ambiguità. | Aprire i documenti indicati e motivare la decisione. |
| Non verificabile | Formato, lettura, provenienza o copertura non consentono una conclusione sul requisito. | Chiarire il limite prima di attribuire un errore allo studente. |
| Non applicabile | Una condizione esplicita rende il requisito non pertinente al caso. | Verificare la condizione, senza trattare l'esito come una mancanza. |
| Proposta linguistica | Il modello propone un giudizio su passaggi citati. | Leggere il testo originale e i passaggi adiacenti, comprese negazioni e condizioni. |
| Coincidenza tra consegne | I componenti o passaggi descritti coincidono secondo una regola dichiarata. | Considerare consegna, fonti comuni e dataset condiviso; nessuna attribuzione automatica di copiatura. |

Completezza, correttezza didattica e somiglianze tra consegne vanno esaminate separatamente. Nessuno di questi campi determina da solo voto, autore, intenzione, direzione degli scambi o origine AI.

## Evidence Engine e significato della confidenza

L'Evidence Engine aggiunge un catalogo strutturato ai rilievi, conservandone il significato. Ogni record può includere identificativo, origine deterministica o linguistica, parte della consegna, stato, testo del rilievo, hash degli artefatti, ancore riscontrate e controlli di supporto. Le citazioni Word valide sono legate alla sorgente originale e alle posizioni dei caratteri. Un hash identifica i byte osservati; non è una firma autenticata dell'autore.

Il campo `confidence.support_score` riguarda esclusivamente la **tracciabilità della fonte**: esprime la quota dei controlli di supporto elencati che risultano soddisfatti. Quando manca un'osservazione citata, il punteggio può essere `null`. Controllare sempre anche `checks`, `missing` e il perimetro `source_traceability_only`.

**Il punteggio non è confidenza semantica, accuratezza del giudizio, probabilità di frode o di origine AI.** `semantic_confidence` resta `null` e la calibrazione è dichiarata non disponibile. Una citazione esatta dimostra che quel testo è presente; il modello può comunque interpretarlo male. I rilievi deterministici privi di ancore puntuali conservano il legame con gli artefatti indicati, senza fingere una localizzazione più precisa.

Le proposte linguistiche discordanti sulla stessa parte restano registrate come contraddizioni irrisolte. Il programma non risolve il conflitto con un voto di maggioranza. La copertura distingue elementi valutati, non verificabili, non applicabili e ancorati; un rapporto annullato mantiene esplicite le sezioni non eseguite.

## Cache di parsing e diagnostica forensic

La cache nasce per ogni singola esecuzione di `audit_zip` e viene eliminata al termine. Riutilizza risultati di lettura riusciti dello stesso contenuto, distinguendo tipo di documento e versione dello schema della cache. Ogni utilizzatore riceve una copia indipendente, così i metadati della consegna corrente non contaminano altri studenti. Gli errori non vengono conservati come risultati validi.

I limiti predefiniti sono 32 voci e **32 MiB di budget contabilizzato**, con sostituzione delle voci meno utilizzate di recente. Il conteggio stima il grafo degli oggetti Python trattenuti, la rappresentazione JSON e un margine per voce. È un'approssimazione delle allocazioni trattenute: non limita la RAM totale del processo, il picco dei documenti in lettura, il rapporto o il modello. Le copie temporanee possono richiedere altra memoria. La cache non è persistente e non velocizza automaticamente una nuova scansione della stessa classe.

I contatori `hits`, `misses`, `bypassed`, `evictions`, `entries` e `bytes` consentono di osservare il comportamento effettivo. Nel caso sintetico da 100 studenti con duplicati si sono osservati 40 hit e 160 miss; nel caso di file tutti distinti, zero hit e 200 miss. La presenza di hit non dimostra da sola un miglioramento del tempo complessivo.

Il JSON forensic raccoglie versione, ambiente, hash dello ZIP, stato e copertura, diagnostica dei confronti, compatibilità e catalogo delle evidenze. L'esportazione applica elenchi espliciti dei campi ammessi: non include configurazioni del motore, credenziali, prompt grezzi o documenti originali completi. Può contenere nomi, identificativi, citazioni e metadati degli studenti; va custodita insieme ai rapporti. È una registrazione diagnostica locale, non una catena di custodia certificata.

## Compatibilità offline e cambiamenti dei formati

Il contratto dei lettori è `omv11-docx-body-1`. L'archivio OMV riconosciuto è `11.0`; il lettore conserva la versione dichiarata e gli errori. Questo riconoscimento non certifica ogni versione dell'applicazione jamovi, ogni modulo aggiuntivo o la freschezza degli output salvati.

Per Word, l'evidenza segue i paragrafi del corpo e delle tabelle nell'ordine XML riconosciuto. Intestazioni, note, forme e altre componenti escluse sono dichiarate nel manifesto di estrazione. Le posizioni XML non equivalgono a pagine Word. I confronti testuali utilizzano le sorgenti nell'ordine originale quando disponibili; limiti su blocchi, token e candidati compaiono nella diagnostica. Un limite raggiunto può ridurre la ricerca e impedisce di interpretare l'assenza di segnalazioni come ricerca esaustiva.

Il controllo di compatibilità è locale: `network_check_performed` è falso e non viene cercata online una versione più recente durante l'analisi. Un futuro cambiamento di jamovi o Word richiederà documenti di prova, aggiornamento del lettore e verifica del contratto. La compatibilità futura non è accertabile dai soli numeri di versione attuali.

## Test automatici e riproduzione delle misure

I test di base rimangono eseguibili senza dipendenze esterne:

```bash
python -S -m unittest discover -s tests -v
```

I casi avversari aggiunti esercitano i limiti aggregati degli archivi, il conteggio delle directory, la compressione elevata, percorsi ostili nei documenti, DTD anche in XML UTF-16, delimitatori di ruolo e HTML, prefissi molto lunghi e template obbligatori condivisi. I controlli UTF-16 includono documenti validi, per verificare che la protezione non respinga indiscriminatamente quella codifica.

Le proprietà generative sono solo di sviluppo. Utilizzare un ambiente virtuale con Python almeno 3.11:

```bash
python -m venv .venv-dev
# Linux: .venv-dev/bin/python; Windows: .venv-dev\Scripts\python.exe
.venv-dev/bin/python -m pip install -r requirements-dev.txt
.venv-dev/bin/python -m unittest discover -s tests -p test_properties.py -v
```

Hypothesis è fissato a `6.170.5` in `requirements-dev.txt`; non è una dipendenza dell'applicazione distribuita. Le sei proprietà riguardano codifiche Word, nomi ZIP, percorsi non sicuri, hash del contenuto, ordine dell'inventario e invarianza delle coppie confrontate. Ogni proprietà ha fino a 60 esempi deterministici. Senza Hypothesis, il modulo viene esplicitamente saltato: uno skip non è una proprietà eseguita. Nella prova locale isolata tutte e sei le proprietà sono state eseguite con esito positivo; i 13 test avversari e di contratto del benchmark sono passati.

Il workflow [`affidabilita.yml`](../.github/workflows/affidabilita.yml) utilizza Linux e Python 3.12, installa solo le dipendenze di sviluppo, esegue questi test e misura classi da 10 e 30 studenti nei due scenari. Controlla struttura e contatori del JSON, poi conserva l'artefatto per 14 giorni. Non scarica né avvia modelli. I risultati effettivi sono quelli del singolo job concluso; l'esistenza del workflow non prova che una futura esecuzione passi.

Per una misura locale più ampia:

```bash
python tools/benchmark_audit.py --sizes 10 100 --seed 1729 --output benchmark_artifacts/benchmark.json
```

Il generatore utilizza ZIP sintetici riproducibili, colonne numeriche, una piccola tabella descrittiva salvata e testo Word con marcatori lessicali artificiali. Lo scenario `mixed_duplicates` ripete una coppia di documenti ogni cinque studenti. Sono ammessi al massimo otto dimensionamenti distinti, da 1 a 1000 studenti, e ZIP sintetici fino a 64 MiB. Questi limiti non significano che sia stato misurato il caso da 1000 studenti.

Il JSON registra seed, hash dell'input e del sorgente, versione software, Python e piattaforma, disponibilità e hash del riferimento locale, durata, picco `tracemalloc`, contatori e rapporti di scala. La durata comprende `audit_zip` e la generazione HTML, escludendo la costruzione della fixture; l'inferenza linguistica è disabilitata. Il picco misura allocazioni Python tracciate e include il costo della strumentazione: **non è RSS**, memoria nativa totale o consumo di Minerva.

## Risultati misurati e regressione residua

Misure locali del 20 ottobre 2026 su Linux x86-64, Python 3.12.14, seed 1729. Ogni riga è una singola esecuzione strumentata, senza inferenza. Le dimensioni seguenti sono MB decimali, cioè byte divisi per 1.000.000.

| Stato del codice | Scenario, 100 studenti | Secondi | Picco tracciato, MB |
|---|---|---:|---:|
| Baseline iniziale | File distinti | 48,50 | 77,41 |
| Baseline iniziale | Duplicati misti | 34,69 | 77,84 |
| Integrazione evidenze, prima della compattazione | File distinti | 59,99 | 187,60 |
| Integrazione evidenze, prima della compattazione | Duplicati misti | 61,32 | 188,11 |
| Dopo la compattazione | File distinti | 57,05 | 111,13 |
| Dopo la compattazione | Duplicati misti | 58,21 | 111,40 |

Fonti delle misure: `uallera-benchmark-initial.json`, `uallera-benchmark-after.json` e `benchmark_compact.json`, prodotti durante il collaudo. Gli hash degli input da 100 studenti coincidono nei tre passaggi. Le misure successive registrano anche disponibilità e digest del riferimento locale; il primo JSON precede l'aggiunta di questi campi. I riferimenti riservati non sono inclusi nella repository pubblica: una misura CI senza riferimento può produrre conteggi e tempi diversi.

La compattazione riduce il picco da circa 188 a 111 MB, circa il 41% in meno rispetto all'integrazione intermedia. Rimane però una regressione rispetto alla baseline da circa 77 MB: il picco finale è circa il 43–44% maggiore. I tempi finali restano circa 57 e 58 secondi, rispetto ai precedenti 48,50 e 34,69. L'estensione conserva più informazioni di evidenza; i dati non dimostrano un miglioramento complessivo delle prestazioni.

Nei due casi finali da 100 studenti sono registrati 37.800 rilievi deterministici. Lo scenario distinto ha zero rilievi incrociati; quello con duplicati ne ha 60: 20 coincidenze testuali, 20 OMV identici e 20 correlazioni temporali accessorie. Questi conteggi descrivono la fixture e i controlli eseguiti, non la sensibilità a frodi reali. Nel campione misto i candidati letterali sono 20 e quelli di blocco quasi coincidente 140, su 4950 coppie possibili.

Le prime misure da 10 e 100 studenti riportano rapporti di scala; la prova compatta è stata ripetuta soltanto a 100 studenti. Non si può attribuirle un nuovo rapporto 10→100. Una singola misura non stima variabilità, percentili o velocità garantite su Windows x64 e ARM64. Prima di fissare un obiettivo operativo servono più ripetizioni, documenti rappresentativi, RSS e consumo del modello sull'hardware effettivo.

## Motori Windows autonomi e limiti dei controlli nativi

Il motore CPU deriva dal commit ufficiale llama.cpp `d80619397c2ba5cf1366c29f7d43699ca3a159b9` (`b11756`). La build integra il runtime MSVC, disattiva OpenMP, HTTPS e librerie condivise. I pacchetti originali dipendevano da DLL Visual C++ non incluse; una prova su un runner già provvisto di tali librerie non dimostrava l’autonomia su un computer nuovo.

Il controllo `tools/inspect_native_imports.py --require-static-runtime` legge gli import PE ordinari e ritardati, verifica architettura e librerie mancanti e respinge dipendenze CRT/OpenMP anche se incluse. Gli API set ammessi sono un elenco finito. Il controllo non verifica esportazioni ABI, ogni chiamata dinamica a `LoadLibrary` o qualsiasi versione futura di Windows. Per questo resta necessaria anche la prova di avvio nativa; nessuno dei due controlli misura la qualità delle risposte Minerva.

La compilazione x64 richiede una CPU con AVX2, FMA, F16C e BMI2. Per ARM64, il compilatore Microsoft richiede `GGML_NATIVE=ON`; i probe opzionali dotprod, i8mm, SVE e SME sono disabilitati esplicitamente per evitare di incorporare le estensioni del runner. La build deve verificare anche l’assenza di FP16 vettoriale opzionale. Le feature effettive, i digest e il comando di provenienza sono negli asset del motore; non si tratta di una certificazione su tutto l’hardware Windows.

La prima build statica x64 (`minerva-b11756-static-1`) superava il controllo degli import e `--version`, ma si interrompeva caricando il modello con errore `0xC000001D`. La diagnostica ha registrato un’istruzione AVX-512 `vmovups zmm0` all’offset relativo `0x600405`; il log della compilazione conferma `/arch:AVX512`, non il profilo AVX2 previsto. Questa build x64 non deve essere distribuita come funzionante. La build ARM64 della stessa release ha invece completato un’inferenza pubblica sintetica e la chiusura del processo nel [run nativo](https://github.com/valerioghezzi/UAllera/actions/runs/39497173500): 415,960 secondi compresi preparazione e avvio. Nel relativo protocollo, l’elaborazione del prompt richiede 270,174 secondi e la generazione 66,248 secondi; sono misure su quel runner, non stime per una classe o per qualsiasi computer ARM64.

La nuova ricetta x64 passa gli argomenti CMake come lista strutturata e verifica insieme cache, valore effettivo di `GGML_NATIVE` e opzioni di ogni configurazione del progetto CPU generato. Un valore scritto nella cache non basta: il controllo respinge AVX-512 e richiede esplicitamente AVX2 e runtime statico. La pubblicazione della nuova release è condizionata anche a una richiesta reale al modello e alla chiusura del processo gestito. Il debugger conserva indirizzo e byte dell’istruzione eseguibile, non dump della memoria o credenziali.

Il [gate della build corretta](https://github.com/valerioghezzi/UAllera/actions/runs/39500479580) è passato: cache e DEBUG riportano `GGML_NATIVE=OFF`, il progetto CPU genera `/arch:AVX2` in quattro configurazioni e l’inferenza con il modello è completa. La release x64 `minerva-b11756-static-2` contiene il pacchetto di 5.038.436 byte con SHA-256 `96a01cba8ac316bc64b514d5ff19270af160bd9ab426c54fc04b3c3a6d729251`; il programma verifica questo digest. La preparazione degli asset ha richiesto 33,593 secondi, l’avvio 10,440 e la richiesta sintetica 40,466 sul runner x64. Il processo gestito è stato chiuso correttamente. Questi tempi distinguono le fasi e non dimostrano accuratezza semantica. ARM64 mantiene l’asset verificato della release precedente, con digest distinto.

L’avvio gestito attende una risposta positiva da `/health` e successivamente l’alias corretto da `/v1/models`, entro un termine complessivo di 600 secondi. Non considera pronto un servizio che pubblicizza il modello mentre è ancora in caricamento. Il riscaldamento iniziale è disabilitato; download e verifica degli asset precedono questo termine e l’inferenza ha il proprio timeout. L’interfaccia propone 900 secondi per operazione linguistica, modificabili dal tutor nel limite ammesso: il precedente valore di 300 secondi era inferiore al tempo della prova ARM64. Queste condizioni non garantiscono una latenza adeguata su hardware con poche risorse.

Le licenze comprendono i file LICENSE/COPYING/NOTICE del sorgente e i notice incorporati dei componenti individuati, compreso Mozilla/llamafile in `sgemm.cpp`. Le build e i loro risultati effettivi sono consultabili nel workflow dedicato; il solo testo della configurazione non dimostra che un pacchetto sia stato compilato correttamente.

## Prova linguistica effettiva e risorse locali

Il profilo `minerva-evidence-2.0-hf-tokenizer` ha completato la prova del 20 ottobre 2026: **13 giudizi concordanti su 70 casi sintetici (18,57%)**, con 70 risposte interpretabili. Nessuno dei 28 errori attesi è stato riconosciuto come errore; 30 risposte sono state rassicuranti quando non previsto. Il basso tasso di allarmi non è evidenza di qualità: il richiamo degli errori è nullo. I 72 input effettivi registrati (70 casi e due Word) coincidono integralmente con il template e il tokenizer ufficiali verificati; questa prova resta distinta dalla precedente codifica errata.

Le due relazioni Word con conclusioni opposte sulla causalità hanno ricevuto entrambe «incompleta» per la parte 6. Il gate tecnico-linguistico è fallito. I 70 casi hanno richiesto 1475,716 secondi su Linux x64; download, caricamento e prova dei Word sono esclusi da questa durata. Gli artefatti del [run effettivo](https://github.com/valerioghezzi/UAllera/actions/runs/39490573473) conservano richieste, risposte, protocolli, rapporti e limiti. Sono casi artificiali redatti dall’assistente, senza validazione esterna; non misurano la sensibilità alle frodi reali. Il programma mostra l’esito negativo e mantiene le proposte linguistiche sperimentali, da riscontrare dal tutor.

Il successivo esperimento `experimental-blue-triangle-calibration-1` aggiunge quattro dimostrazioni astratte soltanto al messaggio SYSTEM del revisore, poi valuta otto nuovi casi bilanciati con criteri normativi completi. Gli esempi e la soglia 8/8 sono fissati prima dell’inferenza. Il [run separato](https://github.com/valerioghezzi/UAllera/actions/runs/39496451448) ha prodotto otto risposte valide ma **zero concordanti**, in 103,618 secondi; sono comparse anche motivazioni sul pacco degli esempi estranei. La calibrazione è respinta e non modifica il profilo applicato agli studenti. È un esperimento con esempi nel prompt, non addestramento dei pesi. Casi, richieste, risposte e gate rimangono disponibili insieme alla baseline dei 70 casi; nessun esito negativo è trasformato in prova di qualità.

Un tentativo locale precedente, nello spazio di lavoro limitato a 8 GiB di memoria, è stato interrotto prima di ottenere giudizi completi. È un limite delle risorse osservate, non una misura della qualità semantica. La prova completa è stata effettuata su un runner Linux diverso; non fornisce tempi o consumi garantiti su Windows x64 o ARM64. L’avvio del binario Windows e la qualità dell’inferenza rimangono verifiche distinte.

## Estensioni a plugin: proposta futura

Una possibile evoluzione è separare adattatori dei formati, regole didattiche e produttori di evidenze mediante contratti versionati. Un registro esplicito potrebbe indicare compatibilità, identificativo del componente, versione, ingressi, uscite e test richiesti. **Questa architettura è una proposta: non esiste un caricatore di plugin implementato.**

Prima dell'adozione occorre definire chi può installare codice, isolamento, limiti di risorse, gestione degli aggiornamenti e compatibilità degli schemi. Codice aggiuntivo può accedere ai documenti, consumare risorse, alterare i giudizi o introdurre dipendenze non incluse nel pacchetto. I file degli studenti non devono diventare sorgenti di estensioni eseguibili. Un nuovo plugin richiederebbe prove indipendenti su casi validi, corrotti e avversari, oltre al controllo della provenienza; non renderebbe automaticamente affidabile l'interpretazione semantica.
