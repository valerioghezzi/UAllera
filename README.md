# UAllera

Controllo locale delle esercitazioni jamovi e delle relazioni Word, raccolte in uno ZIP con cartelle degli studenti. Produce un unico rapporto HTML con anomalie, evidenze e limiti, studente per studente.

## Avvio su Windows x64 e ARM64

1. Estrarre completamente il pacchetto della propria architettura.
2. Aprire `Avvia_UAllera.cmd`: Python è già incluso; non occorrono installazioni di Python, R, Ollama o jamovi.
3. Premere **Prepara e avvia Minerva**. La prima preparazione scarica il modello pubblico ufficiale, circa 4,48 GB, e il motore nativo. I file vengono verificati con dimensione e SHA-256 prefissati. Questo scaricamento non contiene documenti degli studenti.
4. Le esecuzioni successive utilizzano i file locali verificati. Dopo la preparazione, l’analisi può svolgersi senza connessione Internet.
5. Selezionare lo ZIP delle consegne, avviare il controllo e scaricare il rapporto unico.
6. Premere **Termina programma** per chiudere anche il motore locale avviato dall’applicazione.

Il motore x64 richiede AVX2, FMA, F16C e BMI2. Il profilo ARM64 e i limiti delle verifiche native sono descritti nella [guida all’affidabilità](docs/Affidabilita.md).

In alternativa, `Prepara_Minerva.cmd` prepara i file attraverso il runtime già incluso. Il limite dello ZIP delle consegne è 512 MiB. Gli ZIP annidati sono segnalati, senza espansione automatica. I documenti originali restano intatti.

## Minerva e stato della versione

La versione di sviluppo `0.3.0-dev.3` integra **Minerva-7B-Instruct v1.0**, Sapienza/CINECA, nella conversione GGUF ufficiale Q4_K_M. Non è stato effettuato un addestramento aggiuntivo. Il modello interpreta le risposte; il tutor conferma ogni proposta.

Il protocollo `minerva-evidence-2.0-hf-tokenizer` legge i paragrafi del corpo e delle tabelle nell’ordine XML, comprese tabelle annidate e controlli di contenuto riconosciuti. Non campiona i primi 128 paragrafi. Intestazioni, note, forme o altri contenuti non riconosciuti compaiono fra le esclusioni del rapporto.

La finestra è limitata a 4096 token. Il programma controlla il template ufficiale e utilizza il tokenizer ufficiale verificato con SHA-256 per preparare gli identificativi dei token, inviati direttamente all’inferenza. Il tokenizer nativo del GGUF ha prodotto sequenze diverse da quello ufficiale. Il profilo corretto coincide con le sequenze ufficiali su 56 prompt di conformità e su tutti i 72 input effettivi della nuova prova; la qualità linguistica resta insufficiente, come documentato sotto. Il conteggio riserva spazio per la risposta. I passaggi troppo lunghi vengono suddivisi con posizioni dei caratteri conservate. I confini possono separare condizioni o negazioni: il tutor deve verificare i passaggi adiacenti.

Il modello restituisce identificativi dei passaggi. Citazioni e posizioni vengono ricavate dal Word originale dal programma. Una citazione riscontrata prova la presenza del testo, non la correttezza del giudizio. Le osservazioni discordanti restano visibili; nessun voto deriva da maggioranze o dall’assenza di rilievi. Le tabelle OMV sono contesto salvato, non risultati ricalcolati; se non entrano nella finestra, l’esclusione viene dichiarata.

**L’integrazione software e la qualità linguistica sono verifiche distinte.** I test del protocollo non dimostrano che Minerva comprenda correttamente le risposte. La prova reale comprende 70 casi sintetici invariati e due Word con conclusioni diverse sulla causalità. La prima esecuzione Minerva ha prodotto 10 etichette concordanti su 46 tentativi, prima di fermarsi su una risposta non utilizzabile. Il successivo confronto ha riscontrato sequenze di input diverse dal tokenizer ufficiale in tutti i 48 prompt registrati, comprese le due relazioni Word. Quel profilo è inaffidabile e il risultato non isola la capacità del modello con codifica corretta. Gli artefatti della [prima prova](https://github.com/valerioghezzi/UAllera/actions/runs/39484212106) restano conservati come baseline negativa. Consultare i rapporti effettivi del workflow Minerva; una configurazione di collaudo non equivale a un esito positivo.

La [prova con tokenizer corretto](https://github.com/valerioghezzi/UAllera/actions/runs/39490573473) è conclusa: 70 tentativi e 70 risposte interpretabili, **13 etichette concordanti su 70 (18,57%)**, richiamo degli errori **0 su 28**, 30 rassicurazioni errate secondo la griglia del test. Entrambe le relazioni Word hanno ricevuto «incompleta» sulla causalità; la distinzione attesa non è stata rilevata. La valutazione dei 70 casi ha richiesto 1475,716 secondi su Linux x64, senza comprendere preparazione e prova Word. Il collaudo non è superato: le proposte Minerva rimangono sperimentali e non consentono una valutazione autonoma degli studenti. Il corpus è sintetico e redatto dall’assistente; questi valori non stimano prestazioni su una popolazione reale. Gli input e gli esiti attesi non sono stati modificati per migliorare il risultato.

Un [esperimento separato di calibrazione](https://github.com/valerioghezzi/UAllera/actions/runs/39496451448), con quattro esempi astratti del revisore e otto nuovi casi bilanciati fissati prima dell’inferenza, ha ottenuto **0 giudizi concordanti su 8**, tutti formalmente interpretabili (103,618 secondi di valutazione). Il modello ha trasferito anche motivazioni sugli esempi estranei. Questo override resta esclusivamente nel programma sperimentale `tools/probe_minerva_calibration.py` e non viene applicato all’analisi degli studenti. Non è stato effettuato addestramento dei pesi né modificato il corpus originale.

L’annullamento conserva le osservazioni già ottenute in un rapporto incompleto, con numero di studenti e sezioni non esaminati. Una richiesta in corso resta soggetta al timeout locale.

## Controlli dei file e integrità

Il lettore OMV cerca output effettivamente salvati: la sola presenza di un’analisi impostata non basta. Il formato implementato è archivio jamovi 11.0; i formati non riconosciuti sono dichiarati non verificabili. La presenza degli output non certifica che siano aggiornati. I confronti numerici vengono eseguiti dove i dati sono interpretabili in modo univoco.

I controlli incrociati comprendono somiglianze testuali, file OMV identici anche rinominati o ricompressi, metadati disponibili e coincidenze da approfondire. Dati e risultati prescritti comuni non bastano da soli a segnalare copiatura. Gli orari ZIP hanno precisione e fuso limitati; i metadati possono essere modificati. Il programma non determina autore, direzione degli scambi, intenzione, colpevolezza o percentuali di origine AI.

Il report distingue controlli deterministici, proposte linguistiche, evidenze incrociate e requisiti non verificabili. La partecipazione all’esercitazione non è dimostrabile dai soli file.

## Evidenze, cache e diagnostica

L’Evidence Engine associa i rilievi agli hash degli artefatti, ai passaggi riscontrati e alla copertura effettiva. Conserva i conflitti irrisolti. Il punteggio di supporto indica soltanto la tracciabilità della fonte; non misura correttezza del giudizio o probabilità di frode. La confidenza semantica rimane non calibrata.

Una cache in memoria, separata per esecuzione, riutilizza i risultati di lettura di file identici con copie indipendenti. Il budget contabilizzato è 32 MiB e 32 voci; non limita la RAM dell’intera applicazione. Il rapporto mostra hit, miss e limiti. La modalità forensic esporta diagnostica ed evidenze in JSON locale:

```bash
python audit_cli.py consegne.zip -o Rapporto.html --forensic Diagnostica.json
```

Il controllo di compatibilità conserva versioni e componenti non riconosciuti senza consultare servizi online. I dettagli, le misure e i rischi residui sono nella [guida all’affidabilità](docs/Affidabilita.md). L’architettura a plugin è una proposta futura.

Nel benchmark locale su 100 studenti sintetici il picco tracciato è sceso da circa 188 a 111 MB dopo la compattazione, ma resta superiore alla baseline di circa 77 MB. I tempi osservati sono circa 57–58 secondi. Sono singole misure strumentate senza inferenza, non prestazioni garantite. Il workflow di affidabilità misura classi da 10 e 30 studenti e conserva i risultati, senza avviare modelli.

## Sviluppo e collaudi

L’applicazione richiede solo la libreria standard di Python; i pacchetti Windows comprendono un runtime CPython ufficiale verificato. Per lo sviluppo:

```bash
python -S -m unittest discover -s tests -v
python audit_cli.py consegne.zip -o Rapporto_esercitazioni.html
python app.py
```

Il workflow generale esegue i test su Linux, Windows x64 e Windows ARM64, verifica il runtime portatile e il caricamento del motore nativo. Il test di sviluppo Minerva, su Linux x64, scarica soltanto risorse pubbliche e impiega documenti artificiali:

```bash
python tools/ci_minerva.py --output minerva_artifacts
```

Le risposte effettive, i tempi, gli hash, la copertura, gli errori e le proposte sono conservati negli artefatti. La versione Minerva gestita richiede anche il tokenizer ufficiale verificato; il profilo esterno con lo stesso nome non procede senza questo asset. Hypothesis è una dipendenza esclusivamente di sviluppo, fissata in `requirements-dev.txt`; senza di essa le proprietà sono dichiarate saltate. Le soglie del test sono criteri tecnici prestabiliti su casi sintetici redatti dall’assistente, da approvare con il docente; non costituiscono una validazione esterna né una misura della sensibilità alle frodi.

La [precedente prova Qwen](https://github.com/valerioghezzi/UAllera/actions/runs/39466687033) ha ottenuto 15 etichette concordanti su 70 tentativi e non giustificava valutazioni linguistiche affidabili. I risultati restano separati dal nuovo esperimento Minerva. Ollama, LM Studio e llama.cpp esterni rimangono disponibili fra i motori avanzati; il client ammette soltanto indirizzi HTTP di loopback e non può verificare internamente un servizio esterno configurato dal tutor.

La repository pubblica contiene soltanto codice e casi sintetici. I riferimenti privati dell’attività e le consegne reali restano esclusi. Una cartella `reference` locale può aggiungere riferimenti numerici e testi comuni da escludere dai confronti.
