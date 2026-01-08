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
                        // Add one <image> placeholder (PDFs are now concatenated into a single image)
                        buffer.push_str("<image>");
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
    let mut max_width = 0u32;
    let mut total_height = 0u32;

    // Render all pages
    for page_index in 0..document.pages().len() {
        let page = document
            .pages()
            .get(page_index)
            .map_err(|err| ApiError::Internal(format!("failed to get PDF page {}: {err}", page_index)))?;

        // Render at 2x scale (144 DPI) for better quality OCR
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

        max_width = max_width.max(width);
        total_height += height;
        page_images.push(img);
    }

    // Concatenate all pages vertically into a single image
    let mut combined = image::RgbaImage::new(max_width, total_height);
    let mut y_offset = 0u32;

    for page_img in page_images {
        let page_width = page_img.width();
        let page_height = page_img.height();

        // Center the page horizontally if it's narrower than max_width
        let x_offset = (max_width - page_width) / 2;

        image::imageops::overlay(&mut combined, &page_img, x_offset as i64, y_offset as i64);
        y_offset += page_height;
    }

    Ok(vec![DynamicImage::ImageRgba8(combined)])
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
