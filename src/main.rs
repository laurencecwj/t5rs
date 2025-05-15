// Copyright 2019-present, the HuggingFace Inc. team, The Google AI Language Team and Facebook, Inc.
// Copyright 2019 Guillaume Becquin
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//     http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use rust_bert::pipelines::common::{ModelResource, ModelType};
use rust_bert::pipelines::summarization::{SummarizationConfig, SummarizationModel};
use rust_bert::resources::RemoteResource;
use rust_bert::t5::{T5ConfigResources, T5ModelResources, T5VocabResources};

fn test_summary() -> anyhow::Result<()> {
    let config_resource = RemoteResource::from_pretrained(T5ConfigResources::T5_SMALL);
    let vocab_resource = RemoteResource::from_pretrained(T5VocabResources::T5_SMALL);
    let weights_resource = RemoteResource::from_pretrained(T5ModelResources::T5_SMALL);

    let summarization_config = SummarizationConfig::new(
        ModelType::T5,
        ModelResource::Torch(Box::new(weights_resource)),
        config_resource,
        vocab_resource,
        None,
    );
    println!("Model loaded in {:?}", summarization_config.device);
    let summarization_model = SummarizationModel::new(summarization_config)?;

    let input = ["In findings published Tuesday in Cornell University's arXiv by a team of scientists \
from the University of Montreal and a separate report published Wednesday in Nature Astronomy by a team \
from University College London (UCL), the presence of water vapour was confirmed in the atmosphere of K2-18b, \
a planet circling a star in the constellation Leo. This is the first such discovery in a planet in its star's \
habitable zone — not too hot and not too cold for liquid water to exist. The Montreal team, led by Björn Benneke, \
used data from the NASA's Hubble telescope to assess changes in the light coming from K2-18b's star as the planet \
passed between it and Earth. They found that certain wavelengths of light, which are usually absorbed by water, \
weakened when the planet was in the way, indicating not only does K2-18b have an atmosphere, but the atmosphere \
contains water in vapour form. The team from UCL then analyzed the Montreal team's data using their own software \
and confirmed their conclusion. This was not the first time scientists have found signs of water on an exoplanet, \
but previous discoveries were made on planets with high temperatures or other pronounced differences from Earth. \
\"This is the first potentially habitable planet where the temperature is right and where we now know there is water,\" \
said UCL astronomer Angelos Tsiaras. \"It's the best candidate for habitability right now.\" \"It's a good sign\", \
said Ryan Cloutier of the Harvard–Smithsonian Center for Astrophysics, who was not one of either study's authors. \
\"Overall,\" he continued, \"the presence of water in its atmosphere certainly improves the prospect of K2-18b being \
a potentially habitable planet, but further observations will be required to say for sure. \" \
K2-18b was first identified in 2015 by the Kepler space telescope. It is about 110 light-years from Earth and larger \
but less dense. Its star, a red dwarf, is cooler than the Sun, but the planet's orbit is much closer, such that a year \
on K2-18b lasts 33 Earth days. According to The Guardian, astronomers were optimistic that NASA's James Webb space \
telescope — scheduled for launch in 2021 — and the European Space Agency's 2028 ARIEL program, could reveal more \
about exoplanets like K2-18b."];

    let ts = std::time::Instant::now();
    //    Credits: WikiNews, CC BY 2.5 license (https://en.wikinews.org/wiki/Astronomers_find_water_vapour_in_atmosphere_of_exoplanet_K2-18b)
    let _output = summarization_model.summarize(&input)?;
    println!("summarize time: {:?}", ts.elapsed());
    for sentence in _output {
        println!("{sentence}");
    }

    Ok(())
}

fn test_t5_generation() -> anyhow::Result<()> {
    use tch::{nn, Device};
    use rust_bert::resources::{LocalResource, ResourceProvider};
    use rust_bert::Config;
    use rust_bert::t5::{T5Config, T5ForConditionalGeneration, T5Generator};  
    use std::path::Path;
    use rust_bert::pipelines::text_generation::{TextGenerationConfig, TextGenerationModel};
    use std::path::PathBuf;
    use rust_bert::pipelines::common::{ModelType, TokenizerOption};

    let base_dir = "./t5-base";
    let config_path = PathBuf::from(format!("{base_dir}/config.json"));
        
    // Load model
    // let device = Device::cuda_if_available();
    // let mut vs = nn::VarStore::new(device);
    // let config = T5Config::from_file(&config_path);
    // let t5_model = T5ForConditionalGeneration::new(&vs.root(), &config);
    
    // // Load weights (assuming you have model weights in the directory)
    // vs.load(PathBuf::from(format!("{base_dir}/model.ot")))?;
    
    let config = T5Config::from_file(&config_path);

    // let tokenizer = TokenizerOption::from_hf_tokenizer_file(
    //     PathBuf::from(format!("{base_dir}/tokenizer.json")),
    //     None,
    // )?;

    // Input text
    let input_text = "translate English to German: The house is wonderful.";

    
    // Create generator
    let generate_config = TextGenerationConfig {
        model_type: ModelType::T5,
        config_resource: Box::new(LocalResource::from(PathBuf::from(format!("{base_dir}/config.json")))),
        max_length: Some(100),
        do_sample: false,
        num_beams: 5,
        temperature: 0.1,
        num_return_sequences: 1,
        ..Default::default()
    };
    
    let pipeline = TextGenerationModel::new(generate_config)?;
    
    let ts = std::time::Instant::now();
    // Generate output
    let output = pipeline.generate(&[input_text.to_string()], None)?;
    println!("generate time: {:?}", ts.elapsed());
    println!("Translation: {}", output[0]); 
    
    // // Tokenize input
    // let tokenizer = rust_bert::pipelines::common::TokenizerOption::from_file(
    //     PathBuf::from(format!("{base_dir}/tokenizer.json")),
    //     PathBuf::from(format!("{base_dir}/tokenizer_config.json")),
    //     None,
    //     false,
    //     None,
    //     None,
    // )?;
    
    // let input = tokenizer.encode(input_text, true);
    
    // let ts = std::time::Instant::now();
    // // Generate output
    // let output = t5_model.generate(
    //     &[input],
    //     Some(GenerateOptions {
    //         max_length: Some(50),
    //         ..Default::default()
    //     }),
    // )?;
    // println!("generate time: {:?}", ts.elapsed());
    
    // // Decode output
    // let translation = tokenizer.decode(&output[0], true, true)?;
    
    // println!("Translation: {}", translation);
    Ok(())
}


fn main() -> anyhow::Result<()> {
    test_t5_generation()?;
    Ok(())
}