"""Development-only browser check with synthetic files; Playwright is a CI dependency."""
from pathlib import Path
import io
import sys
import threading
import zipfile
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'tests'))
from playwright.sync_api import sync_playwright
import app
from test_pipeline import empty_omv, word


def main():
    server=app.make_server(0)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    data=io.BytesIO()
    with zipfile.ZipFile(data,'w') as z:
        z.writestr('Classe/Uno/a.omv',empty_omv());z.writestr('Classe/Uno/a.docx',word())
        z.writestr('Classe/Due/a.omv',b'danneggiato')
    errors=[];external=[]
    try:
        with sync_playwright() as p:
            browser=p.chromium.launch(channel='chrome',headless=True,args=['--no-sandbox'])
            page=browser.new_page(accept_downloads=True)
            page.on('pageerror',lambda error:errors.append(str(error)))
            def route(request):
                url=request.request.url
                if not url.startswith(('http://127.0.0.1:','blob:')):
                    external.append(url);request.abort()
                else:request.continue_()
            page.route('**/*',route)
            page.goto(f'http://127.0.0.1:{server.server_address[1]}/?token={server.token}')
            page.select_option('#provider','lmstudio')
            assert page.input_value('#endpoint')=='http://127.0.0.1:1234'
            page.select_option('#provider','')
            page.set_input_files('#zip',{'name':'Consegne_artificiali.zip','mimeType':'application/zip','buffer':data.getvalue()})
            page.click('#start')
            page.locator('#download').wait_for(state='visible',timeout=30000)
            with page.expect_download() as downloaded:page.click('#download')
            download=downloaded.value
            assert download.suggested_filename=='Rapporto_esercitazioni.html'
            html=Path(download.path()).read_text(encoding='utf-8')
            assert html.count('id="student-')==2
            assert html.count('<html')==1
            assert not errors,errors
            assert not external,external
            page.select_option('#provider','ollama')
            page.fill('#model','modello-non-installato-per-test')
            page.click('#validate')
            page.locator('#download').wait_for(state='visible',timeout=30000)
            with page.expect_download() as validation_download:page.click('#download')
            checked=validation_download.value
            assert checked.suggested_filename=='Validazione_modello.html'
            validation_html=Path(checked.path()).read_text(encoding='utf-8')
            assert 'Validazione linguistica' in validation_html
            assert 'Non disponibile' in validation_html
            assert not errors,errors
            assert not external,external
            page.click('#exit')
            browser.close()
        print('Browser: ZIP caricato, 2 studenti, rapporto unico scaricato, validazione modello e fallback verificati, nessun errore JS o richiesta esterna.')
    finally:
        server.shutdown();server.server_close();thread.join(timeout=3)


if __name__=='__main__':main()
