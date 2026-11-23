//! WhisperKit (Core ML folder) model downloads.
//!
//! A WhisperKit model is a directory tree fetched from a Hugging Face repo —
//! compiled `.mlmodelc` bundles plus tokenizer files — unlike the single-file
//! GGUF models the rest of the manager handles. File list and sizes come from
//! the HF tree API at download time (nothing but the repo/revision pin is
//! hardcoded); every LFS file carries a sha256 the transfer verifies.

use super::{DiskStatus, DownloadProgress, ModelInfo, ModelManager, ProgressTarget};
use anyhow::{anyhow, Context, Result};
use hf_hub::api::tokio::CancellationToken;
use log::info;
use serde::Deserialize;
use std::fs;
use std::path::{Path, PathBuf};
use tauri::Emitter;

/// Files (relative to the model folder) a complete download must contain.
/// Mirrors the Swift side's `ModelLoader.requiredPaths`.
pub const REQUIRED_FILES: [&str; 5] = [
    "config.json",
    "tokenizer.json",
    "AudioEncoder.mlmodelc/model.mil",
    "MelSpectrogram.mlmodelc/model.mil",
    "TextDecoder.mlmodelc/model.mil",
];

/// Where WhisperKit models live under the app's models dir.
pub fn model_dir(models_dir: &Path, folder: &str) -> PathBuf {
    models_dir.join("whisperkit").join(folder)
}

/// Whether `dir` holds a complete WhisperKit model (required files present).
pub fn is_complete(dir: &Path) -> bool {
    REQUIRED_FILES.iter().all(|file| dir.join(file).is_file())
}

/// Bytes of `*.partial` files inside a model dir, for resume display.
pub fn partial_bytes(dir: &Path) -> u64 {
    fn visit(dir: &Path, out: &mut u64) {
        for entry in fs::read_dir(dir).into_iter().flatten().flatten() {
            let path = entry.path();
            if path.is_dir() {
                visit(&path, out);
            } else if path.extension().is_some_and(|e| e == "partial") {
                *out += entry.metadata().map(|m| m.len()).unwrap_or(0);
            }
        }
    }
    let mut total = 0;
    visit(dir, &mut total);
    total
}

/// One file of the model folder, from the HF tree API.
#[derive(Debug, Clone, Deserialize)]
struct TreeFile {
    path: String,
    #[serde(rename = "type")]
    kind: String,
    size: u64,
    /// Present for LFS-backed files; `oid` is their sha256.
    lfs: Option<LfsInfo>,
}

#[derive(Debug, Clone, Deserialize)]
struct LfsInfo {
    oid: String,
}

/// The repo-relative files under `folder`, with sizes and (for LFS) sha256s.
async fn list_repo_files(
    repo_id: &str,
    revision: &str,
    folder: &str,
) -> Result<Vec<(String, u64, Option<String>)>> {
    let url = format!(
        "https://huggingface.co/api/models/{repo_id}/tree/{revision}?recursive=true&expand=true"
    );
    let response = fetch_tree_json(&url)
        .await
        .context("listing WhisperKit model files on Hugging Face")?;
    let tree: Vec<TreeFile> =
        serde_json::from_str(&response).context("parsing the Hugging Face file tree")?;
    let prefix = format!("{folder}/");
    Ok(tree
        .into_iter()
        .filter(|file| file.kind == "file" && file.path.starts_with(&prefix))
        .map(|file| {
            let relative = file.path[prefix.len()..].to_string();
            (relative, file.size, file.lfs.map(|lfs| lfs.oid))
        })
        .collect())
}

/// GET a JSON document with a small retry ladder (HF rate limits and
/// transient 5xx both surface as errors otherwise).
async fn fetch_tree_json(url: &str) -> Result<String> {
    let client = reqwest::Client::builder()
        .user_agent("Handy (github.com/cjpais/Handy)")
        .build()?;
    let mut last_error = String::new();
    for _ in 0..3 {
        match client.get(url).send().await {
            Ok(response) if response.status().is_success() => {
                return response.text().await.context("reading the file tree");
            }
            Ok(response) => {
                last_error = format!("HTTP {}", response.status());
            }
            Err(e) => last_error = e.to_string(),
        }
        tokio::time::sleep(std::time::Duration::from_secs(2)).await;
    }
    Err(anyhow!("fetching {url}: {last_error}"))
}

impl ModelManager {
    /// Download a WhisperKit folder model, file by file, into the app's
    /// models dir. Registry bookkeeping and events live here; the transfer
    /// itself is [`download_whisperkit_folder`], which needs no app handle.
    pub(super) async fn download_whisperkit_model(
        &self,
        model_info: &ModelInfo,
        repo_id: &str,
        revision: &str,
        folder: &str,
    ) -> Result<()> {
        let model_id = model_info.id.clone();
        let dir = model_dir(&self.models_dir, folder);

        if is_complete(&dir) {
            // Nothing to do; clear stale partials from an interrupted run.
            remove_partials(&dir);
            self.update_download_status()?;
            return Ok(());
        }

        {
            let mut models = self.available_models.lock().unwrap();
            if let Some(model) = models.get_mut(&model_id) {
                model.is_downloading = true;
            }
        }
        let cancel_token = CancellationToken::new();
        {
            let mut flags = self.cancel_flags.lock().unwrap();
            flags.insert(model_id.clone(), cancel_token.clone());
        }
        let mut cleanup = super::DownloadCleanup {
            available_models: &self.available_models,
            cancel_flags: &self.cancel_flags,
            model_id: model_id.clone(),
            disarmed: false,
        };

        let progress_handle = self.app_handle.clone();
        let event_handle = self.app_handle.clone();
        let event_id = model_id.clone();
        let outcome = download_whisperkit_folder(
            &self.models_dir,
            &model_id,
            repo_id,
            revision,
            folder,
            &cancel_token,
            &move |progress| {
                let _ = progress_handle.emit("model-download-progress", &progress);
            },
            &move |event| {
                let _ = event_handle.emit(event, &event_id);
            },
        )
        .await;

        match outcome {
            Ok(FolderOutcome::Completed) => {
                cleanup.disarmed = true;
                {
                    let mut models = self.available_models.lock().unwrap();
                    if let Some(model) = models.get_mut(&model_id) {
                        model.is_downloading = false;
                        model.is_downloaded = true;
                        model.partial_size = 0;
                    }
                }
                self.cancel_flags.lock().unwrap().remove(&model_id);
                let _ = self.app_handle.emit("model-download-complete", &model_id);
                info!("WhisperKit model downloaded: {}", model_id);
                Ok(())
            }
            Ok(FolderOutcome::Cancelled) => {
                info!("WhisperKit model download cancelled: {}", model_id);
                let _ = self.app_handle.emit("model-download-cancelled", &model_id);
                Ok(())
            }
            Err(e) => Err(e),
        }
    }

    /// Delete a WhisperKit model folder. Fails if it is the active model —
    /// callers unload first (same contract as the single-file delete).
    pub(super) fn delete_whisperkit_model(
        &self,
        model_info: &ModelInfo,
        folder: &str,
    ) -> Result<()> {
        let dir = model_dir(&self.models_dir, folder);
        if !dir.exists() {
            return Ok(());
        }
        fs::remove_dir_all(&dir).with_context(|| format!("removing {}", dir.display()))?;
        // Drop the (now empty) whisperkit root too, so the models dir looks
        // pristine in the UI's folder view.
        let root = self.models_dir.join("whisperkit");
        if let Ok(entries) = fs::read_dir(&root) {
            if entries.count() == 0 {
                let _ = fs::remove_dir(&root);
            }
        }
        info!("Deleted WhisperKit model: {}", model_info.id);
        Ok(())
    }
}

/// How a folder download ended; cancellation keeps partials without erroring.
pub(super) enum FolderOutcome {
    Completed,
    Cancelled,
}

/// The transport: list the repo folder, fetch each file with resume and
/// sha256 verification, and validate the result. Free-standing so tests can
/// drive it without an app handle.
#[allow(clippy::too_many_arguments)]
pub(super) async fn download_whisperkit_folder(
    models_dir: &Path,
    model_id: &str,
    repo_id: &str,
    revision: &str,
    folder: &str,
    cancel_token: &CancellationToken,
    on_progress: &(dyn Fn(DownloadProgress) + Send + Sync),
    on_event: &(dyn Fn(&str) + Send + Sync),
) -> Result<FolderOutcome> {
    use super::download::HttpDownloadEvent;
    use super::download::HttpDownloadOutcome;

    let dir = model_dir(models_dir, folder);
    let files = list_repo_files(repo_id, revision, folder).await?;
    if files.is_empty() {
        return Err(anyhow!(
            "Hugging Face repo {repo_id} contains no files under '{folder}'"
        ));
    }
    let total: u64 = files.iter().map(|(_, size, _)| size).sum();
    info!(
        "Downloading WhisperKit model '{}' from {repo_id}@{revision}: {} files, {:.1} MB",
        model_id,
        files.len(),
        total as f64 / 1_000_000.0
    );

    // Bytes already finished; each file's progress is offset by them so the
    // whole folder fills one bar.
    let mut done: u64 = 0;
    for (relative, size, sha256) in &files {
        let target_file = dir.join(relative);
        if target_file.is_file()
            && matches!(fs::metadata(&target_file), Ok(meta) if meta.len() == *size)
        {
            // Complete from an earlier attempt.
            done += size;
            continue;
        }
        fs::create_dir_all(target_file.parent().expect("file path has a parent"))?;
        let partial = dir.join(format!("{relative}.partial"));
        let url =
            format!("https://huggingface.co/{repo_id}/resolve/{revision}/{folder}/{relative}");
        let target = ProgressTarget {
            model_id: model_id.to_string(),
            offset: done,
            total: Some(done + size),
        };
        let progress_target = target.clone();
        let outcome = ModelManager::download_http_resumable_with_events(
            model_id,
            &url,
            &partial,
            Some(*size),
            sha256.as_deref(),
            cancel_token,
            &move |event| match event {
                HttpDownloadEvent::Progress(progress) => on_progress(progress_target.progress(
                    progress.downloaded,
                    progress.total,
                    progress.bytes_per_sec,
                )),
                HttpDownloadEvent::VerificationStarted => on_event("model-verification-started"),
                HttpDownloadEvent::VerificationCompleted => {
                    on_event("model-verification-completed")
                }
            },
        )
        .await?;
        match outcome {
            HttpDownloadOutcome::Cancelled => return Ok(FolderOutcome::Cancelled),
            HttpDownloadOutcome::Completed => {}
        }
        fs::rename(&partial, &target_file).with_context(|| format!("finalizing {relative}"))?;
        done += size;
    }

    if !is_complete(&dir) {
        return Err(anyhow!(
            "model folder is incomplete after downloading every listed file"
        ));
    }
    Ok(FolderOutcome::Completed)
}

/// Remove every `*.partial` inside `dir` (used when the folder is already
/// complete).
fn remove_partials(dir: &Path) {
    fn visit(dir: &Path) {
        for entry in fs::read_dir(dir).into_iter().flatten().flatten() {
            let path = entry.path();
            if path.is_dir() {
                visit(&path);
            } else if path.extension().is_some_and(|e| e == "partial") {
                let _ = fs::remove_file(&path);
            }
        }
    }
    if dir.is_dir() {
        visit(dir);
    }
}

/// Reused by the display path: on-disk status of a WhisperKit model.
pub fn disk_status(models_dir: &Path, folder: &str) -> DiskStatus {
    let dir = model_dir(models_dir, folder);
    DiskStatus {
        is_downloaded: is_complete(&dir),
        is_downloading: false,
        partial_size: if dir.is_dir() { partial_bytes(&dir) } else { 0 },
    }
}

#[cfg(all(test, target_os = "macos", target_arch = "aarch64"))]
mod e2e_tests {
    use super::*;

    const REPO: &str = "omnixima/bengaliai-whisper-medium-coreml";
    const REVISION: &str = "8b5433d62c5841deb0055c7eedfe60e10021e7cc";
    const FOLDER: &str = "bengaliai-whisper-medium";
    const MODEL_ID: &str = "whisperkit/bengaliai-whisper-medium";

    /// The real app's models dir, so the model this downloads is the one the
    /// real app (and `--transcribe-file`) finds.
    fn real_models_dir() -> PathBuf {
        let home = std::env::var("HOME").expect("$HOME");
        let dir = PathBuf::from(home).join("Library/Application Support/com.pais.handy/models");
        std::fs::create_dir_all(&dir).expect("create models dir");
        dir
    }

    /// Full pipeline: repo listing, multi-file download with verification,
    /// folder validation, Swift engine load, silence decode, unload.
    /// Ignored: downloads ~584 MB; run with
    /// `cargo test download_loads_and_transcribes -- --ignored --nocapture`.
    #[tokio::test]
    #[ignore = "downloads a ~584 MB model; run explicitly"]
    async fn download_loads_and_transcribes() {
        let models_dir = real_models_dir();
        let token = CancellationToken::new();
        let outcome = download_whisperkit_folder(
            &models_dir,
            MODEL_ID,
            REPO,
            REVISION,
            FOLDER,
            &token,
            &|_| {},
            &|_| {},
        )
        .await
        .expect("download");
        assert!(matches!(outcome, FolderOutcome::Completed));

        let dir = model_dir(&models_dir, FOLDER);
        assert!(is_complete(&dir), "downloaded folder is complete");

        let engine = crate::whisperkit::WhisperKitEngine::new();
        engine.load_model(&dir).expect("model loads through FFI");
        assert!(engine.is_model_loaded());
        // 1 s of silence: must succeed; WhisperKit may emit nothing at all.
        let silence = vec![0.0f32; 16_000];
        let quiet = engine
            .transcribe(&silence, Some("bn"))
            .expect("silence decodes");
        assert!(
            quiet.trim().len() < 16,
            "silence yields no real text: {quiet:?}"
        );
        engine.unload_model();
        assert!(!engine.is_model_loaded(), "unload frees the model");
    }

    /// A re-run over a complete folder is a no-op that still verifies.
    #[tokio::test]
    #[ignore = "needs the model downloaded by download_loads_and_transcribes"]
    async fn redownload_is_idempotent() {
        let models_dir = real_models_dir();
        let token = CancellationToken::new();
        let outcome = download_whisperkit_folder(
            &models_dir,
            MODEL_ID,
            REPO,
            REVISION,
            FOLDER,
            &token,
            &|_| {},
            &|_| {},
        )
        .await
        .expect("re-download");
        assert!(matches!(outcome, FolderOutcome::Completed));
    }

    /// Cancellation across the boundary: a cancel racing a decode either
    /// fails with Cancelled or completes; never panics, never leaks.
    #[test]
    #[ignore = "needs the model downloaded by download_loads_and_transcribes"]
    fn cancel_is_cooperative() {
        let models_dir = real_models_dir();
        let engine = crate::whisperkit::WhisperKitEngine::new();
        engine
            .load_model(&model_dir(&models_dir, FOLDER))
            .expect("load");
        let samples = vec![0.05f32; 16_000 * 5];
        engine.cancel();
        match engine.transcribe(&samples, Some("bn")) {
            Err(crate::whisperkit::WhisperKitError::Cancelled) => {}
            Ok(_) => {}
            other => panic!("cancel-before-run must cancel or complete: {other:?}"),
        }
        engine.unload_model();
    }
}
