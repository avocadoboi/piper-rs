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
    let synthesizer = PiperSpeechSynthesizer::new(Arc::new(VitsModel::new("sv_SE-nst-medium.onnx.json".into(), "sv_SE-nst-medium.onnx".into(), &ENVIRONMENT)?))?;
    synthesizer.synthesize_to_file("kaniner.wav", "Kaniner är små och fluffiga.".to_string())?;

    Ok(())
}
