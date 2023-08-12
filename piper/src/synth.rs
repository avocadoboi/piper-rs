use std::collections::vec_deque::VecDeque;
use std::io::Cursor;
use std::sync::Arc;

use once_cell::sync::{Lazy, OnceCell};

use rayon::prelude::*;
use rayon::{ThreadPool, ThreadPoolBuilder};

use crate::core::{PiperError, PiperModel, PiperResult, PiperWaveResult, PiperWaveSamples};
use crate::wave_writer;

//----------------------------------------------------------------

// #[allow(dead_code)]
// pub fn param_to_percent(value: f32, min: f32, max: f32) -> u8 {
//     ((value - min) / (max - min) * 100.).round() as u8
// }

// pub fn percent_to_param(value: u8, min: f32, max: f32) -> f32 {
//     value as f32 / 100. * (max - min) + min
// }

/// Batch size when using batched synthesis mode
const SPEECH_STREAM_BATCH_SIZE: usize = 4;

static SYNTHESIS_THREAD_POOL: Lazy<ThreadPool> = Lazy::new(|| {
    ThreadPoolBuilder::new()
        .thread_name(|i| format!("piper_synth_{}", i))
        .num_threads(num_cpus::get())
        .build()
        .unwrap()
});

pub struct PiperSpeechSynthesizer(Arc<dyn PiperModel + Sync + Send>);

impl PiperSpeechSynthesizer {
    pub fn new(model: Arc<dyn PiperModel + Sync + Send>) -> PiperResult<Self> {
        Ok(Self(model))
    }

    fn create_synthesis_task_provider(
        &self,
        text: String,
    ) -> SpeechSynthesisTaskProvider {
        SpeechSynthesisTaskProvider {
            model: Arc::clone(&self.0),
            text,
        }
    }

    pub fn synthesize_lazy(
        &self,
        text: String,
    ) -> PiperResult<PiperSpeechStreamLazy> {
        PiperSpeechStreamLazy::new(self.create_synthesis_task_provider(text))
    }
    pub fn synthesize_parallel(
        &self,
        text: String,
    ) -> PiperResult<PiperSpeechStreamParallel> {
        PiperSpeechStreamParallel::new(self.create_synthesis_task_provider(text))
    }
    pub fn synthesize_batched(
        &self,
        text: String,
        batch_size: Option<usize>,
    ) -> PiperResult<PiperSpeechStreamBatched> {
        let mut batch_size = batch_size.unwrap_or(SPEECH_STREAM_BATCH_SIZE);
        if batch_size == 0 {
            batch_size = SPEECH_STREAM_BATCH_SIZE;
        }
        PiperSpeechStreamBatched::new(
            self.create_synthesis_task_provider(text),
            batch_size,
        )
    }

    pub fn synthesize_to_samples(&self, text: String) -> PiperResult<Vec<i16>> {
        if text.is_empty() {
            return Ok(vec![]);
        }

        let mut samples: Vec<i16> = Vec::new();
        for result in self.synthesize_parallel(text)? {
            match result {
                Ok(ws) => {
                    samples.append(&mut ws.to_vec());
                }
                Err(e) => return Err(e),
            };
        }
        
        if samples.is_empty() {
            return Err(PiperError::OperationError(
                "No speech data to write".to_string(),
            ));
        }

        Ok(samples)
    }
    pub fn synthesize_to_wav_buffer(&self, text: String) -> PiperResult<Vec<u8>> {
        let samples = self.synthesize_to_samples(text)?;
        
        let mut wave_buffer = Vec::new();
        wave_writer::write_wave_samples_to_buffer(
            Cursor::new(&mut wave_buffer), 
            samples.iter(), 
            self.0.wave_info()?.sample_rate as u32,
            self.0.wave_info()?.num_channels.try_into().unwrap(),
            self.0.wave_info()?.sample_width.try_into().unwrap(),
        )?;
        Ok(wave_buffer)
    }
    pub fn synthesize_to_wav_file(&self, filename: &str, text: String) -> PiperResult<()> {
        let samples = self.synthesize_to_samples(text)?;
        Ok(wave_writer::write_wave_samples_to_file(
            filename.into(),
            samples.iter(),
            self.0.wave_info()?.sample_rate as u32,
            self.0.wave_info()?.num_channels.try_into().unwrap(),
            self.0.wave_info()?.sample_width.try_into().unwrap(),
        )?)
    }
}

struct SpeechSynthesisTaskProvider {
    model: Arc<dyn PiperModel + Sync + Send>,
    text: String,
}

impl SpeechSynthesisTaskProvider {
    fn get_phonemes(&self) -> PiperResult<Vec<String>> {
        Ok(self.model.phonemize_text(&self.text)?.to_vec())
    }
    fn process_one_sentence(&self, phonemes: String) -> PiperWaveResult {
        self.model.speak_one_sentence(phonemes)
    }
    #[allow(dead_code)]
    fn process_batches(&self, phonemes: Vec<String>) -> PiperResult<Vec<PiperWaveSamples>> {
        self.model.speak_batch(phonemes)
    }
}

pub struct PiperSpeechStreamLazy {
    provider: SpeechSynthesisTaskProvider,
    sentence_phonemes: std::vec::IntoIter<String>,
}

impl PiperSpeechStreamLazy {
    fn new(provider: SpeechSynthesisTaskProvider) -> PiperResult<Self> {
        let sentence_phonemes = provider.get_phonemes()?.into_iter();
        Ok(Self {
            provider,
            sentence_phonemes,
        })
    }
}

impl Iterator for PiperSpeechStreamLazy {
    type Item = PiperWaveResult;

    fn next(&mut self) -> Option<Self::Item> {
        let next_batch = self.sentence_phonemes.next()?;
        match self.provider.process_one_sentence(next_batch) {
            Ok(ws) => Some(Ok(ws)),
            Err(e) => Some(Err(e)),
        }
    }
}

#[must_use]
pub struct PiperSpeechStreamParallel {
    precalculated_results: std::vec::IntoIter<PiperWaveResult>,
}

impl PiperSpeechStreamParallel {
    fn new(provider: SpeechSynthesisTaskProvider) -> PiperResult<Self> {
        let calculated_result: Vec<PiperWaveResult> = provider
            .get_phonemes()?
            .par_iter()
            .map(|ph| provider.process_one_sentence(ph.to_string()))
            .collect();
        Ok(Self {
            precalculated_results: calculated_result.into_iter(),
        })
    }
}

impl Iterator for PiperSpeechStreamParallel {
    type Item = PiperWaveResult;

    fn next(&mut self) -> Option<Self::Item> {
        self.precalculated_results.next()
    }
}

#[must_use]
pub struct PiperSpeechStreamBatched {
    provider: Arc<SpeechSynthesisTaskProvider>,
    sentence_phonemes: std::vec::IntoIter<String>,
    channel: SpeechSynthesisChannel,
    batch_size: usize,
}

impl PiperSpeechStreamBatched {
    fn new(provider: SpeechSynthesisTaskProvider, batch_size: usize) -> PiperResult<Self> {
        let sentence_phonemes = provider.get_phonemes()?.into_iter();
        let mut instance = Self {
            provider: Arc::new(provider),
            sentence_phonemes,
            channel: SpeechSynthesisChannel::new(batch_size)?,
            batch_size,
        };
        instance.send_batch();
        Ok(instance)
    }
    fn send_batch(&mut self) {
        let next_batch = Vec::from_iter((&mut self.sentence_phonemes).take(self.batch_size));
        if !next_batch.is_empty() {
            let provider = Arc::clone(&self.provider);
            self.channel.put(provider, next_batch);
        }
    }
}

impl Iterator for PiperSpeechStreamBatched {
    type Item = PiperWaveResult;

    fn next(&mut self) -> Option<Self::Item> {
        self.send_batch();
        self.channel.get()
    }
}

struct SpeechSynthesisTask(Arc<OnceCell<PiperWaveResult>>);

impl SpeechSynthesisTask {
    fn new(provider: Arc<SpeechSynthesisTaskProvider>, phonemes: String) -> Self {
        let instance = Self(Arc::new(OnceCell::new()));
        let result = Arc::clone(&instance.0);
        SYNTHESIS_THREAD_POOL.spawn_fifo(move || {
            let wave_result = provider.process_one_sentence(phonemes);
            result.set(wave_result).unwrap();
        });
        instance
    }
    fn get_result(self) -> PiperWaveResult {
        self.0.wait();
        if let Ok(result) = Arc::try_unwrap(self.0) {
            result.into_inner().unwrap()
        } else {
            Err(PiperError::OperationError(
                "Failed to obtain results".to_string(),
            ))
        }
    }
}

struct SpeechSynthesisChannel {
    task_queue: VecDeque<SpeechSynthesisTask>,
}

impl SpeechSynthesisChannel {
    fn new(batch_size: usize) -> PiperResult<Self> {
        Ok(Self {
            task_queue: VecDeque::with_capacity(batch_size * 4),
        })
    }
    fn put(&mut self, provider: Arc<SpeechSynthesisTaskProvider>, batch: Vec<String>) {
        for phonemes in batch.into_iter() {
            self.task_queue
                .push_back(SpeechSynthesisTask::new(Arc::clone(&provider), phonemes));
        }
    }
    fn get(&mut self) -> Option<PiperWaveResult> {
        self.task_queue.pop_front().map(|task| task.get_result())
    }
}
