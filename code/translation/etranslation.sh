#!/bin/zsh
# Traduzione eTranslation del dataset unified, in sottofondo (si può chiudere il terminale).
# Si lancia dalla cartella del progetto, con sh o zsh:
#
#   sh code/translation/etranslation.sh start     verifica le credenziali e avvia tunnel + traduzione
#   sh code/translation/etranslation.sh restart   ferma tutto e riavvia (riprende da dove era arrivato)
#   sh code/translation/etranslation.sh status    avanzamento, ultima traduzione arrivata, stima del tempo
#   sh code/translation/etranslation.sh log       ultime righe del log, dal vivo (Ctrl+C esce, il lavoro continua)
#   sh code/translation/etranslation.sh stop      ferma tutto
#   sh code/translation/etranslation.sh forget    cancella la password salvata nel Portachiavi
#
# Se il tunnel cade, o eTranslation smette di rispondere, il lanciatore riapre il tunnel
# (nuovo indirizzo) e riparte da solo. Si ferma solo a lavoro finito, con stop, o se la
# password viene rifiutata. La password sta nel Portachiavi del Mac, mai in un file.

# con "sh" il file viene rieseguito da zsh (la sintassi sotto è di zsh)
if [ -z "$ZSH_VERSION" ]; then exec zsh "$0" "$@"; fi

SELF="${0:A}"                     # percorso di questo file (dentro le funzioni $0 è il nome della funzione)
cd "${SELF:h}/../.." || exit 1      # cartella del progetto (questo file sta in code/translation/)
PORT=8765
LOGDIR=results/logs/etranslation
ENDPOINT=https://language-tools.ec.europa.eu/etranslation/api/askTranslate
MOD=code/translation/etranslation.py
KC=etranslation                     # nome della voce nel Portachiavi
SPEED=(--pause 0.1 --max-inflight 50)
mkdir -p $LOGDIR

ts() { date '+%d/%m %H:%M' }
running() { pgrep -f "etranslation.sh _loop" >/dev/null }

stop_all() {
  pkill -f "etranslation.sh _loop"
  pkill -f "$MOD run"
  pkill -f "cloudflared tunnel --url http://127.0.0.1:$PORT"
  sleep 2
}

# stessa prova della guida (punto 6): una frase, callback finti; serve solo il codice HTTP
test_credentials() {
  curl -s -o /dev/null -w '%{http_code}' -u "$ETR_USER:$ETR_PASS" \
    -H 'Content-Type: application/json' -X POST $ENDPOINT \
    -d '{"callerInformation":{"externalReference":"credential-check"},"textToTranslate":"This is a test.","sourceLanguage":"EN","targetLanguages":["IT"],"domain":"GEN","notifications":{"success":{"http":"https://example.com/success/"},"failure":{"http":"https://example.com/failure/"}},"deliveries":{"http":"https://example.com/delivery/"}}'
}

get_credentials() {
  [[ -n "${ETR_USER:-}" && -n "${ETR_PASS:-}" ]] && return
  local acct
  acct=$(security find-generic-password -s $KC 2>/dev/null | awk -F'"' '/"acct"/{print $4}')
  if [[ -n $acct ]]; then
    ETR_USER=$acct
    ETR_PASS=$(security find-generic-password -s $KC -w 2>/dev/null)
    FROM_KC=1
    echo "Credenziali prese dal Portachiavi (application name: $ETR_USER)."
    return
  fi
  while true; do
    read "ETR_USER?Application name: "
    read -s "ETR_PASS?Password: "; echo
    ETR_USER=${ETR_USER// /}
    if (( ${#ETR_PASS} < 6 )); then
      echo "La password ricevuta è lunga ${#ETR_PASS} caratteri: sembra non incollata bene, riprova."
    else
      break
    fi
  done
}

loop() {
  local first=1 extra
  while true; do
    pkill -f "cloudflared tunnel --url http://127.0.0.1:$PORT"; sleep 1
    : > $LOGDIR/tunnel.log
    cloudflared tunnel --url http://127.0.0.1:$PORT >> $LOGDIR/tunnel.log 2>&1 &
    TPID=$!
    URL=""
    for i in {1..60}; do
      URL=$(grep -Eo 'https://[a-z0-9-]+\.trycloudflare\.com' $LOGDIR/tunnel.log | grep -v '^https://api\.' | head -1)
      [[ -n $URL ]] && break
      sleep 2
    done
    if [[ -z $URL ]]; then
      echo "$(ts) il tunnel non è partito, riprovo tra 60 s"
      kill $TPID 2>/dev/null; sleep 60; continue
    fi
    echo "$(ts) tunnel aperto: $URL"
    sleep 15                                   # il nuovo indirizzo impiega qualche secondo a essere visibile
    extra=(); (( first )) && extra=(--retry-errors); first=0
    uv run --with requests --with truststore --with tqdm python $MOD run \
      --public-url $URL --requeue-sent $SPEED $extra >> $LOGDIR/run.log 2>&1
    RC=$?
    kill $TPID 2>/dev/null
    case $RC in
      0)   echo "$(ts) TRADUZIONE COMPLETATA"; break ;;
      2)   echo "$(ts) CREDENZIALI RIFIUTATE: mi fermo. Rilancia con: sh code/translation/etranslation.sh start"; break ;;
      130|143) echo "$(ts) fermato"; break ;;
      3)   echo "$(ts) tunnel caduto o eTranslation non risponde: riapro il tunnel tra 30 s"; sleep 30 ;;
      *)   echo "$(ts) uscito con codice $RC, riparto tra 60 s"; sleep 60 ;;
    esac
  done
}

do_start() {
  if running; then echo "È già in corso. Per riavviarlo: sh code/translation/etranslation.sh restart"; exit 0; fi
  command -v cloudflared >/dev/null || { echo "Manca cloudflared: brew install cloudflared"; exit 1; }
  command -v uv >/dev/null || { echo "Manca uv nel PATH di questo terminale"; exit 1; }
  FROM_KC=0
  get_credentials
  echo "Verifico le credenziali con eTranslation..."
  CODE=$(test_credentials)
  case $CODE in
    200) echo "Credenziali accettate." ;;
    401)
      echo "Credenziali RIFIUTATE (HTTP 401) per l'application name [$ETR_USER], password di ${#ETR_PASS} caratteri."
      (( FROM_KC )) && echo "La password salvata non vale più: cancellala con  sh code/translation/etranslation.sh forget  e rilancia start."
      exit 1 ;;
    000) echo "eTranslation non raggiungibile: controlla la connessione e riprova."; exit 1 ;;
    *)   echo "Risposta inattesa di eTranslation (HTTP $CODE): parto lo stesso, lo script gestisce gli errori." ;;
  esac
  if (( ! FROM_KC )) && [[ $CODE == 200 ]]; then
    read "SAVE?Salvo la password nel Portachiavi, così non dovrai più scriverla? [s/n] "
    if [[ $SAVE == [sSyY]* ]]; then
      security add-generic-password -U -a "$ETR_USER" -s $KC -w "$ETR_PASS" && echo "Salvata."
    fi
  fi
  stop_all
  export ETR_USER ETR_PASS
  echo "\n===== avvio $(ts) =====" >> $LOGDIR/run.log
  echo "$(ts) avvio" >> $LOGDIR/supervisor.log
  nohup caffeinate -is zsh "$SELF" _loop >> $LOGDIR/supervisor.log 2>&1 &
  disown
  unset ETR_PASS
  echo "Apro il tunnel..."
  for i in {1..30}; do
    tail -1 $LOGDIR/supervisor.log | grep -q 'tunnel aperto' && break
    sleep 2
  done
  if ! running; then
    echo "\nERRORE: il lanciatore non è partito. Ultime righe del log:"
    tail -5 $LOGDIR/supervisor.log
    exit 1
  fi
  tail -1 $LOGDIR/supervisor.log
  echo "\nAvviato in sottofondo: puoi chiudere il terminale."
  echo "Il Mac deve restare acceso, alla corrente e con lo schermo aperto."
  echo "Avanzamento: sh code/translation/etranslation.sh status"
}

case "${1:-}" in
  start)   do_start ;;
  restart) stop_all; echo "Fermato. Riavvio..."; do_start ;;
  _loop)   loop ;;
  stop)
    stop_all
    echo "$(ts) fermato a mano" >> $LOGDIR/supervisor.log
    echo "Fermato. Le traduzioni arrivate sono salvate; si riprende con: sh code/translation/etranslation.sh start" ;;
  status)
    if running; then echo "STATO: in corso"; else echo "STATO: FERMO (riavvia con: sh code/translation/etranslation.sh start)"; fi
    echo "--- ultimi eventi:"; tail -4 $LOGDIR/supervisor.log 2>/dev/null
    echo "--- ultime righe del programma:"
    tail -c 4000 $LOGDIR/run.log 2>/dev/null | tr '\r' '\n' | grep -v '^ *$' | tail -4
    uv run python $MOD status | sed -n '/Segmenti per stato/,/^$/p;/Prima traduzione/,$p' ;;
  log)
    tail -n 30 -f $LOGDIR/supervisor.log $LOGDIR/run.log ;;
  forget)
    security delete-generic-password -s $KC >/dev/null 2>&1 && echo "Password cancellata dal Portachiavi." \
      || echo "Nessuna password salvata." ;;
  *)
    sed -n '2,15p' "$SELF" ;;
esac
