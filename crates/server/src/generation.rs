use std::{convert::TryFrom, sync::Arc};

use base64::Engine;
use deepseek_ocr_core::{DecodeOutcome, DecodeParameters, ModelKind, VisionSettings};
use image::DynamicImage;
use pdfium_render::prelude::*;
use reqwest::blocking::Client;
use rocket::tokio;
use tokenizers::Tokenizer;
use tracing::info;

use crate::{
    error::ApiError,
    models::{ApiMessage, ImagePayload, MessageContent, MessagePart},
    state::{GenerationInputs, SharedModel},
    stream::{StreamContext, StreamController},
};

type StreamCallback = Box<dyn Fn(usize, &[i64])>;

#[derive(Debug)]
pub struct GenerationResult {
    pub text: String,
    pub prompt_tokens: usize,
    pub response_tokens: usize,
}

pub fn base_decode_parameters(
    inputs: &GenerationInputs,
    max_new_tokens: usize,
) -> DecodeParameters {
    let mut params = inputs.defaults.clone();
    params.max_new_tokens = max_new_tokens;
    params
}

pub async fn generate_async(
    inputs: GenerationInputs,
    prompt: String,
    images: Vec<DynamicImage>,
    params: DecodeParameters,
    stream: Option<StreamContext>,
) -> Result<GenerationResult, ApiError> {
    // If there are multiple images (e.g., multi-page PDF), process each separately
    if images.len() > 1 {
        return generate_multipage_async(inputs, prompt, images, params, stream).await;
    }

    let stream_for_block = stream.clone();
    let join_result = tokio::task::spawn_blocking(move || {
        generate_blocking(
            &inputs.model,
            Arc::clone(&inputs.tokenizer),
            prompt,
            images,
            inputs.vision,
            params,
            stream_for_block,
        )
    })
    .await;

    match join_result {
        Ok(Ok(result)) => Ok(result),
        Ok(Err(err)) => {
            if let Some(ctx) = stream {
                ctx.send_error(&err.to_string());
            }
            Err(err)
        }
        Err(err) => {
            let api_err = ApiError::Internal(format!("generation task failed: {err}"));
            if let Some(ctx) = stream {
                ctx.send_error(&api_err.to_string());
            }
            Err(api_err)
        }
    }
}

async fn generate_multipage_async(
    inputs: GenerationInputs,
    prompt: String,
    images: Vec<DynamicImage>,
    params: DecodeParameters,
    stream: Option<StreamContext>,
) -> Result<GenerationResult, ApiError> {
    use crate::stream::StreamController;

    let mut combined_text = String::new();
    let mut total_prompt_tokens = 0;
    let mut total_response_tokens = 0;

    // For multi-page PDFs, we process each page sequentially without per-page streaming
    // to avoid complexity with multiple finalization events
    for (page_num, image) in images.into_iter().enumerate() {
        let page_inputs = inputs.clone();
        let page_prompt = prompt.replace("<image><image>", "<image>")
            .replacen("<image>", "<image>", 1); // Keep only first <image> placeholder
        let page_params = params.clone();

        let join_result = tokio::task::spawn_blocking(move || {
            generate_blocking(
                &page_inputs.model,
                Arc::clone(&page_inputs.tokenizer),
                page_prompt,
                vec![image],
                page_inputs.vision,
                page_params,
                None, // No per-page streaming for multi-page documents
            )
        })
        .await;

        let result = match join_result {
            Ok(Ok(r)) => r,
            Ok(Err(err)) => {
                if let Some(ref ctx) = stream {
                    ctx.send_error(&err.to_string());
                }
                return Err(err);
            }
            Err(err) => {
                let api_err = ApiError::Internal(format!("page {} generation failed: {err}", page_num + 1));
                if let Some(ref ctx) = stream {
                    ctx.send_error(&api_err.to_string());
                }
                return Err(api_err);
            }
        };

        // Add page separator before subsequent pages
        if !combined_text.is_empty() {
            combined_text.push_str("\n\n<--- Page Split --->\n\n");
        }

        combined_text.push_str(&result.text);
        total_prompt_tokens += result.prompt_tokens;
        total_response_tokens += result.response_tokens;
    }

    // If streaming was requested, send the complete result as a streamed fallback
    if let Some(ctx) = stream {
        let controller = StreamController::new(Arc::clone(&inputs.tokenizer), ctx);
        controller.send_initial();
        controller.emit_fallback(&combined_text);
    }

    Ok(GenerationResult {
        text: combined_text,
        prompt_tokens: total_prompt_tokens,
        response_tokens: total_response_tokens,
    })
}

fn generate_blocking(
    model: &SharedModel,
    tokenizer: Arc<Tokenizer>,
    prompt: String,
    images: Vec<DynamicImage>,
    vision: VisionSettings,
    params: DecodeParameters,
    stream: Option<StreamContext>,
) -> Result<GenerationResult, ApiError> {
    let guard = model
        .lock()
        .map_err(|_| ApiError::Internal("model lock poisoned".into()))?;
    let tokenizer_ref = tokenizer.as_ref();
    let stream_controller = stream.map(|ctx| StreamController::new(Arc::clone(&tokenizer), ctx));
    let mut callback_box: Option<StreamCallback> = None;
    if let Some(controller) = stream_controller.as_ref() {
        controller.send_initial();
        let callback = controller.callback();
        callback_box = Some(Box::new(callback));
    }

    let decode_result = guard.decode(
        tokenizer_ref,
        &prompt,
        &images,
        vision,
        &params,
        callback_box.as_deref(),
    );
    drop(callback_box);

    let outcome = match decode_result {
        Ok(output) => output,
        Err(err) => {
            drop(guard);
            let message = err.to_string();
            if message.contains("prompt formatting failed")
                || message.contains("prompt/image embedding mismatch")
            {
                return Err(ApiError::BadRequest(message));
            }
            return Err(ApiError::Internal(format!("generation failed: {err:#}")));
        }
    };

    drop(guard);

    let DecodeOutcome {
        text: normalized,
        prompt_tokens,
        response_tokens,
        generated_tokens,
    } = outcome;

    let decoded = tokenizer_ref
        .decode(
            &generated_tokens
                .iter()
                .filter_map(|&id| u32::try_from(id).ok())
                .collect::<Vec<_>>(),
            true,
        )
        .unwrap_or_default();

    info!(
        "[generate] decoded_raw=\"{}\" normalized=\"{}\"",
        decoded
            .replace('\n', "\\n")
            .chars()
            .take(120)
            .collect::<String>(),
        normalized
            .replace('\n', "\\n")
            .chars()
            .take(120)
            .collect::<String>()
    );

    if let Some(controller) = stream_controller.as_ref() {
        controller.flush_remaining(&generated_tokens);
        controller.finalize(&normalized, prompt_tokens, response_tokens);
    }

    Ok(GenerationResult {
        text: normalized,
        prompt_tokens,
        response_tokens,
    })
}

pub fn convert_messages(
    kind: ModelKind,
    messages: &[ApiMessage],
) -> Result<(String, Vec<DynamicImage>), ApiError> {
    match kind {
        ModelKind::Deepseek => convert_deepseek_messages(messages),
        ModelKind::PaddleOcrVl | ModelKind::DotsOcr => convert_paddle_messages(messages),
    }
}

fn convert_deepseek_messages(
    messages: &[ApiMessage],
) -> Result<(String, Vec<DynamicImage>), ApiError> {
    let (sections, images) = collect_prompt_sections(messages)?;
    let mut prompt = String::from("");
    let body = sections.join("\n\n").trim().to_owned();
    prompt.push_str(&body);

    Ok((prompt, images))
}

fn convert_paddle_messages(
    messages: &[ApiMessage],
) -> Result<(String, Vec<DynamicImage>), ApiError> {
    let (sections, images) = collect_prompt_sections(messages)?;
    let prompt = sections.join("\n\n").trim().to_owned();
    Ok((prompt, images))
}

fn collect_prompt_sections(
    messages: &[ApiMessage],
) -> Result<(Vec<String>, Vec<DynamicImage>), ApiError> {
    let latest_user_idx = messages
        .iter()
        .rposition(|message| message.role.eq_ignore_ascii_case("user"))
        .ok_or_else(|| {
            ApiError::BadRequest("request must include at least one user message".into())
        })?;

    let mut sections = Vec::new();
    let mut all_images = Vec::new();

    // OCR模型不是为对话训练的，所以只保留一轮的prompt，留多轮连正常输出都产生不了
    for message in &messages[..latest_user_idx] {
        if !message.role.eq_ignore_ascii_case("system") {
            continue;
        }
        let (text, mut msg_images) = flatten_content(&message.content)?;
        if !text.is_empty() {
            sections.push(text);
        }
        all_images.append(&mut msg_images);
    }

    let (user_text, mut user_images) = flatten_content(&messages[latest_user_idx].content)?;
    if !user_text.is_empty() {
        sections.push(user_text);
    }
    all_images.append(&mut user_images);

    if sections.is_empty() && all_images.is_empty() {
        return Err(ApiError::BadRequest(
            "user content must include text or images".into(),
        ));
    }

    if sections.is_empty() && all_images.is_empty() {
        return Err(ApiError::BadRequest(
            "user content must include text or images".into(),
        ));
    }

    Ok((sections, all_images))
}

fn flatten_content(content: &MessageContent) -> Result<(String, Vec<DynamicImage>), ApiError> {
    match content {
        MessageContent::Text(text) => Ok((text.trim().to_owned(), Vec::new())),
        MessageContent::Parts(parts) => {
            let mut buffer = String::new();
            let mut images = Vec::new();
            for part in parts.iter().rev() {
                match part {
                    MessagePart::ImageUrl { image_url } | MessagePart::InputImage { image_url } => {
                        let mut loaded_images = load_image_or_pdf(image_url)?;
                        // Add <image> placeholder for each image/page
                        for _ in 0..loaded_images.len() {
                            buffer.push_str("<image>");
                        }
                        images.append(&mut loaded_images);
                    }
                    MessagePart::Text { text } | MessagePart::InputText { text } => {
                        if !buffer.is_empty() {
                            buffer.push('\n');
                        }
                        buffer.push_str(text);
                    }
                }
            }
            Ok((buffer.trim().to_owned(), images))
        }
    }
}

fn load_image_or_pdf(spec: &ImagePayload) -> Result<Vec<DynamicImage>, ApiError> {
    let url = spec.url();
    if let Some(rest) = url.strip_prefix("data:") {
        return load_data_url_or_pdf(rest);
    }
    if url.starts_with("http://") || url.starts_with("https://") {
        return fetch_remote_image_or_pdf(url);
    }
    Err(ApiError::BadRequest(
        "only data: URIs or http(s) image/PDF URLs are supported".into(),
    ))
}

fn convert_pdf_to_images(pdf_data: &[u8]) -> Result<Vec<DynamicImage>, ApiError> {
    let pdfium = Pdfium::new(
        Pdfium::bind_to_system_library()
            .map_err(|err| ApiError::Internal(format!("failed to initialize PDF library: {err}")))?,
    );

    let document = pdfium
        .load_pdf_from_byte_slice(pdf_data, None)
        .map_err(|err| ApiError::BadRequest(format!("failed to load PDF: {err}")))?;

    if document.pages().len() == 0 {
        return Err(ApiError::BadRequest("PDF contains no pages".into()));
    }

    let mut page_images = Vec::new();

    // Render each page separately at 144 DPI (matching Python implementation)
    for page_index in 0..document.pages().len() {
        let page = document
            .pages()
            .get(page_index)
            .map_err(|err| ApiError::Internal(format!("failed to get PDF page {}: {err}", page_index)))?;

        // 144 DPI = 2x zoom from 72 DPI base
        let render_config = PdfRenderConfig::new()
            .set_target_width(2000)
            .set_maximum_width(4000)
            .rotate_if_landscape(PdfPageRenderRotation::None, true);

        let bitmap = page
            .render_with_config(&render_config)
            .map_err(|err| ApiError::Internal(format!("failed to render PDF page {}: {err}", page_index)))?;

        let width = bitmap.width() as u32;
        let height = bitmap.height() as u32;

        // Convert RGBA bitmap to DynamicImage
        let buffer = bitmap.as_raw_bytes().to_vec();
        let img = image::RgbaImage::from_raw(width, height, buffer)
            .ok_or_else(|| ApiError::Internal(format!("failed to create image from PDF page {}", page_index)))?;

        page_images.push(DynamicImage::ImageRgba8(img));
    }

    Ok(page_images)
}

fn load_data_url_or_pdf(data: &str) -> Result<Vec<DynamicImage>, ApiError> {
    let (meta, payload) = data
        .split_once(',')
        .ok_or_else(|| ApiError::BadRequest("invalid data URL".into()))?;
    if !meta.ends_with(";base64") {
        return Err(ApiError::BadRequest(
            "data URLs must specify base64 encoding".into(),
        ));
    }
    let decoded = base64::engine::general_purpose::STANDARD
        .decode(payload)
        .map_err(|err| ApiError::BadRequest(format!("invalid base64 payload: {err}")))?;

    // Check if it's a PDF by looking at the MIME type or magic bytes
    let is_pdf = meta.starts_with("application/pdf") || decoded.starts_with(b"%PDF");

    if is_pdf {
        // Convert PDF pages to images
        convert_pdf_to_images(&decoded)
    } else {
        // Load as a single image
        let img = image::load_from_memory(&decoded)
            .map_err(|err| ApiError::BadRequest(format!("failed to decode inline image: {err}")))?;
        Ok(vec![img])
    }
}

fn fetch_remote_image_or_pdf(url: &str) -> Result<Vec<DynamicImage>, ApiError> {
    let client = Client::new();
    let response = client
        .get(url)
        .send()
        .map_err(|err| ApiError::BadRequest(format!("failed to fetch {url}: {err}")))?
        .error_for_status()
        .map_err(|err| ApiError::BadRequest(format!("request failed for {url}: {err}")))?;

    // Check content type
    let content_type = response
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string())
        .unwrap_or_default();

    let bytes = response
        .bytes()
        .map_err(|err| ApiError::BadRequest(format!("failed to read response body: {err}")))?;

    let is_pdf = content_type.contains("application/pdf") || bytes.starts_with(b"%PDF");

    if is_pdf {
        // Convert PDF pages to images
        convert_pdf_to_images(&bytes)
    } else {
        // Load as a single image
        let img = image::load_from_memory(&bytes)
            .map_err(|err| ApiError::BadRequest(format!("failed to decode remote image: {err}")))?;
        Ok(vec![img])
    }
}
