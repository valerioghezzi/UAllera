"""Execute the real UI handlers against a minimal DOM and a local-status response."""
import json
import shutil
import subprocess
import unittest
import app


@unittest.skipUnless(shutil.which('node'), 'Development UI state check requires Node; browser CI also tests the UI')
class UiStateTests(unittest.TestCase):
    def test_changing_provider_during_a_job_recovers_model_listing_after_completion(self):
        source = app.PAGE.split('<script>', 1)[1].split('</script>', 1)[0]
        test = r'''
import {readFileSync} from 'node:fs';
import vm from 'node:vm';
import assert from 'node:assert/strict';
const elements = new Map();
const get = id => {
  if (!elements.has(id)) elements.set(id, {value:'',disabled:true,files:[],hidden:false});
  return elements.get(id);
};
const context=vm.createContext({
  document:{getElementById:get}, location:{search:'?token=test'}, URLSearchParams,
  clearInterval:()=>{},
  fetch:async()=>({ok:true,json:async()=>({busy:false,total:1,done:1,message:'Terminato',ready:false})})
});
vm.runInContext(JSON.parse(readFileSync(0,'utf8')),context);
get('provider').value='ollama';
vm.runInContext("busy=true; $('provider').onchange()",context);
assert.equal(get('loadmodels').disabled,true);
await vm.runInContext('poll()',context);
assert.equal(get('loadmodels').disabled,false,'Model listing must recover when the job ends');
'''
        result = subprocess.run([shutil.which('node'), '--input-type=module', '-e', test],
                                input=json.dumps(source), text=True, capture_output=True, timeout=15)
        self.assertEqual(0, result.returncode, result.stderr)
