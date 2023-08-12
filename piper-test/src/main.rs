use std::{
    sync::Arc, 
    error::Error
};

use once_cell::sync::Lazy;

use piper::{
    synth::PiperSpeechSynthesizer,
    vits::VitsModel
};

static ENVIRONMENT: Lazy<Arc<ort::Environment>> = Lazy::new(|| Arc::new(ort::Environment::default()));

fn main() -> Result<(), Box<dyn Error>> {
    let speaker = Arc::new(VitsModel::new("sv_SE-nst-medium.onnx.json".into(), "sv_SE-nst-medium.onnx".into(), &ENVIRONMENT)?);
    for speaker in speaker.speakers()? {
        println!("Speaker {}: {}", speaker.0, speaker.1);
    }
    speaker.set_length_scale(1.)?;
    // speaker.set_speaker("default".to_string())?;
    // speaker.set_speaker("whisper".to_string())?;
    
    let synthesizer = PiperSpeechSynthesizer::new(speaker)?;
    synthesizer.synthesize_to_wav_file("kaniner.wav", "Kaniner är små och fluffiga.".to_string())?;

    Ok(())
}
