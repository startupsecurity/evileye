use anyhow::Result;
use fuzzy_matcher::skim::SkimMatcherV2;
use fuzzy_matcher::FuzzyMatcher;
use regex::Regex;
use std::collections::{HashSet, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use ocrs::{ImageSource, OcrEngine, OcrEngineParams};
use rten::Model;
#[allow(unused)]
use rten_tensor::prelude::*;

struct Args {
    root_path: String,
}

#[derive(Debug)]
struct Img {
    path: String,
    text: String,
    has_secrets: bool,
}

fn parse_args() -> Result<Args, lexopt::Error> {
    use lexopt::prelude::*;

    let mut values = VecDeque::new();
    let mut parser = lexopt::Parser::from_env();

    while let Some(arg) = parser.next()? {
        match arg {
            Value(val) => values.push_back(val.string()?),
            Long("help") => {
                println!(
                    "Usage: {bin_name} <root_path>",
                    bin_name = parser.bin_name().unwrap_or("evileye")
                );
                std::process::exit(0);
            }
            _ => return Err(arg.unexpected()),
        }
    }

    let root_path = values.pop_front().ok_or("missing `root_path` arg")?;

    Ok(Args { root_path })
}

fn file_path(path: &str) -> PathBuf {
    let mut abs_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    abs_path.push(path);
    abs_path
}

async fn find_images_in_directory_concurrent(dir: &Path) -> Result<Vec<PathBuf>> {
    let image_extensions = ["jpg", "jpeg", "png", "gif", "bmp", "tiff"]
        .iter()
        .map(|&s| s.to_string())
        .collect::<HashSet<String>>();
    let mut images = Vec::new();

    let dir = dir.to_path_buf(); // Clone the dir path
    let entries = tokio::task::spawn_blocking(move || {
        walkdir::WalkDir::new(&dir) // Use the cloned dir
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| e.path().is_file())
            .collect::<Vec<_>>()
    })
    .await?;

    let (tx, mut rx) = tokio::sync::mpsc::channel(entries.len());

    for entry in entries {
        let tx = tx.clone();
        let image_extensions = image_extensions.clone();
        tokio::spawn(async move {
            let path = entry.path();
            if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
                if image_extensions.contains(&ext.to_lowercase()) {
                    tx.send(path.to_path_buf())
                        .await
                        .expect("Failed to send path");
                }
            }
        });
    }

    drop(tx);

    while let Some(path) = rx.recv().await {
        images.push(path);
    }

    Ok(images)
}
fn detect_secrets(data: &str) -> bool {
    // TODO pull from more standard list
    let patterns = vec![
        Regex::new(r"AKIA[0-9A-Z]{16}").unwrap(),
        Regex::new(r"(?i)token\s*[:=]\s*\S+").unwrap(),
        Regex::new(r"(?i)password\s*[:=]\s*\S+").unwrap(),
        Regex::new(r"npm_[a-zA-Z0-9@]+").unwrap(),
    ];

    let matcher = SkimMatcherV2::default();

    for pattern in patterns {
        for cap in pattern.captures_iter(data) {
            if let Some(matched) = cap.get(0) {
                let matched_str = matched.as_str();
                if matcher.fuzzy_match(matched_str, matched_str).unwrap_or(0) > 70 {
                    return true;
                }
            }
        }
    }

    false
}

async fn process_image_with_ocr(
    found_image: PathBuf,
    engine: Arc<OcrEngine>,
) -> Result<Img, anyhow::Error> {
    println!("Scanning {}", found_image.display());
    let img = image::open(found_image.clone())?.into_rgb8(); // Clone `found_image` here
    let img_source = ImageSource::from_bytes(img.as_raw(), img.dimensions())?;
    let ocr_input = engine.prepare_input(img_source)?;

    let word_rects = engine.detect_words(&ocr_input)?;
    let line_rects = engine.find_text_lines(&ocr_input, &word_rects);
    let line_texts = engine.recognize_text(&ocr_input, &line_rects)?;

    let lines = line_texts
        .iter()
        .flatten()
        .filter(|l| l.to_string().len() > 1)
        .map(|l| l.to_string())
        .collect::<Vec<String>>();

    Ok(Img {
        path: found_image.to_string_lossy().to_string(), // `found_image` is still available for use
        text: lines.join("\n"),
        has_secrets: false,
    })
}

#[tokio::main]
async fn main() -> Result<()> {
    // Reference: https://github.com/robertknight/ocrs/blob/main/ocrs/examples/hello_ocr.rs
    // Use the `download-models.sh` script to download the models.

    let args = parse_args()?;
    let system_root = Path::new(&args.root_path);

    println!("Running evileye from {}", system_root.display());

    let found_images = find_images_in_directory_concurrent(system_root).await?;
    let detection_model_path = file_path("./text-detection.rten");
    let rec_model_path = file_path("./text-recognition.rten");

    let detection_model = Model::load_file(detection_model_path)?;
    let recognition_model = Model::load_file(rec_model_path)?;

    let engine = Arc::new(OcrEngine::new(OcrEngineParams {
        detection_model: Some(detection_model),
        recognition_model: Some(recognition_model),
        ..Default::default()
    })?);

    println!("Number of found images: {}", found_images.len());

    let image_futures = found_images.into_iter().map(|found_image| {
        let engine = Arc::clone(&engine);
        tokio::spawn(async move { process_image_with_ocr(found_image, engine).await })
    });

    let mut results = futures::future::try_join_all(image_futures).await?;
    for img in &mut results {
        if let Ok(img) = img {
            img.has_secrets = detect_secrets(&img.text);

            println!("-----------------------------------");
            println!("Image Path: {}", img.path);
            println!("Extracted Text:\n{}", img.text);
            println!("Contains Secrets: {}", img.has_secrets);
            println!("-----------------------------------");
        } else {
            println!("-----------------------------------");
            println!("Error processing image: {:?}", img);
            println!("-----------------------------------");
        }
    }
    Ok(())
}
