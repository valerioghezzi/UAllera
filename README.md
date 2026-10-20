# UAllera

Controllo locale di esercitazioni jamovi e relazioni Word contenute in uno ZIP con cartelle degli studenti. Un unico rapporto HTML riporta la situazione di ogni studente, con requisiti, evidenze, discrepanze e verifiche da approfondire.

Il programma non invia i documenti a servizi esterni. L'interfaccia si apre nel browser del PC e comunica soltanto con il processo locale su 127.0.0.1. Il browser può essere utilizzato senza connessione Internet.

## Utilizzo

1. Estrai completamente il pacchetto adatto al PC, Windows x64 oppure Windows ARM64.
2. Apri `Avvia_UAllera.cmd`. Non occorre installare Python, R o jamovi.
3. Seleziona lo ZIP delle cartelle degli studenti e premi **Avvia controllo**.
4. Scarica `Rapporto_esercitazioni.html` e aprilo nel browser. Il rapporto è autonomo e stampabile.
5. Premi **Termina programma** prima di chiudere la pagina.

Il limite dello ZIP è 512 MiB. Le cartelle non vengono appiattite; file mancanti o versioni multiple rimangono visibili nel rapporto. Gli archivi ZIP di consegne annidati dentro lo ZIP principale sono segnalati e non espansi automaticamente.

## Controlli e limiti

La presenza di un'analisi impostata non basta: il lettore cerca tabelle e immagini effettivamente salvate. I controlli numerici sono eseguiti dove il contenuto è interpretabile in modo univoco. Le interpretazioni aperte richiedono il tutor; la scheda mostra domanda, risposta e criterio di verifica.

Le somiglianze e l'identità dei file sono evidenze da contestualizzare. Il programma non determina l'autore, non ricostruisce la direzione degli scambi e non attribuisce percentuali di origine AI o colpevolezza.

Formato OMV implementato: versione archivio 11.0. Per i formati non riconosciuti il rapporto dichiara il limite. La presenza degli output non ne certifica da sola l'aggiornamento: quando non si può accertarlo compare una verifica separata.

## Sviluppo e test

Python 3.13 o successivo, sola libreria standard. Nessun `pip install` per eseguire il programma o i test.

```bash
python -S -m unittest discover -s tests -v
python audit_cli.py consegne.zip -o Rapporto_esercitazioni.html
python app.py
```

Il workflow GitHub esegue i test su Linux, Windows x64 e Windows ARM64 e verifica l'architettura dell'interprete. Un workflow predisposto non equivale a un test passato: consultare gli esiti effettivi dei run.

La repository pubblica contiene soltanto codice e casi artificiali. Le consegne reali, la soluzione completa e i riferimenti privati dell'attività rimangono sul PC. Una cartella `reference` locale può aggiungere i riferimenti numerici e i testi da escludere dai confronti.

## Interpretazione linguistica locale

La funzione facoltativa usa Ollama, LM Studio oppure llama.cpp già avviati sul PC. Nell'interfaccia scegli il motore, premi **Elenca modelli locali**, seleziona il modello e avvia il controllo. Sono ammessi solo indirizzi HTTP di loopback; il programma rifiuta proxy, redirect e modelli dichiarati cloud o remoti. Non richiede chiavi API né acquista crediti. Per garantire che anche il motore scelto resti sul PC, carica prima il modello e svolgi l'analisi senza connessione Internet.

Le rubriche italiane distinguono risposte corrette, errori concettuali, parafrasi, negazioni, mancanze e istruzioni inserite nella relazione. Le proposte riportano citazioni riscontrate nel Word e restano separate dai controlli deterministici. Se il motore è spento, la valutazione linguistica risulta non disponibile e il rapporto dei file viene comunque prodotto.

Non sono inclusi pesi addestrati o un rilevatore di testi AI. La preparazione linguistica consiste in rubriche, esempi e un corpus di calibrazione; la qualità va misurata sul modello concretamente installato. I test del protocollo non misurano l'accuratezza del modello.

Documentazione dei protocolli: [Ollama](https://docs.ollama.com/api/chat), [LM Studio](https://lmstudio.ai/docs/developer/openai-compat/chat-completions), [llama.cpp](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md).

Esempio CLI:

```bash
python audit_cli.py consegne.zip --local-engine ollama --model nome-modello-locale -o Rapporto_esercitazioni.html
```
