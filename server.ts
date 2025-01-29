

import { HfInference } from "@huggingface/inference";
import 'dotenv/config';
import fs from 'node:fs/promises';

const HF_TOKEN = process.env.HF_TOKEN

async function transform() {

  try {
    console.time()

    const text = `
        O rato roeu a ropa do rei de Roma
        O sapo saltou do saco
        Se sacudiu e sumiu da soma

        Tatu, tamanduá, tejubina
        Transaram umas trovas
        Tolices totais
        O vento da vida ventou
        E varreu você pro nunca mais

        O gato do mato, passando prato
        Ai que barato
        Cachorro pedindo socorro
        Na rampa do morro
        E gritando que é zorro

        Um pinto muito distinto
        Tomando abissinto
        Em pleno recinto, e o rastro do vento
        Puxando pra dentro, do meio de centro
        Do nunca mais
    `

    const inference = new HfInference(HF_TOKEN);
    const out = await inference.textToSpeech({
      model: "facebook/mms-tts-por",
      inputs: text,
    });

    const data = await out.arrayBuffer()
    await fs.appendFile('out.wav', Buffer.from(data));
  }
  catch (e) {
    console.log(e)
  } finally {
    console.timeEnd()
  }

}

transform().then().finally(() => process.exit(0))

