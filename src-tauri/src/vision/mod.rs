pub mod preprocessing;
pub mod feature_extraction;
pub mod multimodal_fusion;

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};
use tauri::{AppHandle, Emitter};
use tracing::{info, error};
use image::{DynamicImage, GenericImageView};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CameraFrame {
    pub width: u32,
    pub height: u32,
    pub data: Vec<u8>, // RGB data
    pub timestamp: u64,
}

/// Enhanced vision processing structures for multimodal AI
#[derive(Debug, Clone)]
pub struct VisionConfig {
    pub max_image_size: u32,
    pub target_resolution: (u32, u32),
    pub supported_formats: Vec<String>,
    pub preprocessing_quality: PreprocessingQuality,
    pub enable_face_detection: bool,
    pub enable_object_detection: bool,
}

#[derive(Debug, Clone)]
pub enum PreprocessingQuality {
    Fast,      // Basic resizing and normalization
    Balanced,  // Standard preprocessing with some enhancements
    Quality,   // Full preprocessing pipeline with all enhancements
}

#[derive(Debug, Clone)]
pub struct ImageFeatures {
    pub tensor: Vec<f32>,
    pub width: u32,
    pub height: u32,
    pub channels: u32,
    pub metadata: ImageMetadata,
}

#[derive(Debug, Clone)]
pub struct ImageMetadata {
    pub format: String,
    pub file_size: Option<usize>,
    pub has_faces: bool,
    pub detected_objects: Vec<DetectedObject>,
    pub color_profile: ColorProfile,
}

#[derive(Debug, Clone)]
pub struct DetectedObject {
    pub class: String,
    pub confidence: f32,
    pub bbox: BoundingBox,
}

#[derive(Debug, Clone)]
pub struct BoundingBox {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

#[derive(Debug, Clone)]
pub struct ColorProfile {
    pub dominant_colors: Vec<[u8; 3]>,
    pub brightness: f32,
    pub contrast: f32,
    pub saturation: f32,
}

impl Default for VisionConfig {
    fn default() -> Self {
        Self {
            max_image_size: 1024 * 1024 * 10, // 10MB max
            target_resolution: (224, 224),     // Standard vision model input
            supported_formats: vec![
                "jpg".to_string(),
                "jpeg".to_string(),
                "png".to_string(),
                "gif".to_string(),
                "webp".to_string(),
                "bmp".to_string(),
                "tiff".to_string(),
            ],
            preprocessing_quality: PreprocessingQuality::Balanced,
            enable_face_detection: false,  // Disable by default for privacy
            enable_object_detection: true,
        }
    }
}

/// Enhanced vision processor for multimodal AI
pub struct VisionProcessor {
    config: VisionConfig,
    feature_extractor: feature_extraction::FeatureExtractor,
}

impl VisionProcessor {
    pub fn new(config: VisionConfig) -> Result<Self> {
        info!("Initializing vision processor with config: {:?}", config);
        
        let feature_extractor = feature_extraction::FeatureExtractor::new(&config)?;
        
        Ok(Self {
            config,
            feature_extractor,
        })
    }

    pub fn new_default() -> Result<Self> {
        Self::new(VisionConfig::default())
    }

    /// Process a single image from raw bytes for multimodal AI
    pub fn process_image(&self, image_data: &[u8]) -> Result<ImageFeatures> {
        info!("Processing image ({} bytes)", image_data.len());
        
        // Validate file size
        if image_data.len() > self.config.max_image_size as usize {
            return Err(anyhow::anyhow!(
                "Image too large: {} bytes (max: {} bytes)", 
                image_data.len(), 
                self.config.max_image_size
            ));
        }

        // Decode the image
        let img = image::load_from_memory(image_data)
            .map_err(|e| anyhow::anyhow!("Failed to decode image: {}", e))?;

        // Detect format
        let format = self.detect_image_format(image_data)?;
        
        // Validate format
        if !self.config.supported_formats.contains(&format.to_lowercase()) {
            return Err(anyhow::anyhow!("Unsupported image format: {}", format));
        }

        // Process the image
        self.process_decoded_image(img, format, Some(image_data.len()))
    }

    /// Convert camera frame to image features for AI processing
    pub fn process_camera_frame(&self, frame: &CameraFrame) -> Result<ImageFeatures> {
        info!("Processing camera frame ({}x{})", frame.width, frame.height);
        
        // Convert RGB data to DynamicImage
        let img = image::RgbImage::from_raw(frame.width, frame.height, frame.data.clone())
            .ok_or_else(|| anyhow::anyhow!("Failed to create image from camera data"))?;
        
        let dynamic_img = DynamicImage::ImageRgb8(img);
        
        // Process the image
        self.process_decoded_image(dynamic_img, "rgb".to_string(), None)
    }

    /// Process a decoded image
    fn process_decoded_image(
        &self, 
        img: DynamicImage, 
        format: String, 
        file_size: Option<usize>
    ) -> Result<ImageFeatures> {
        let (original_width, original_height) = img.dimensions();
        info!("Original image dimensions: {}x{}", original_width, original_height);

        // Apply preprocessing based on quality setting
        let processed_img = match self.config.preprocessing_quality {
            PreprocessingQuality::Fast => {
                preprocessing::fast_preprocess(&img, self.config.target_resolution)?
            }
            PreprocessingQuality::Balanced => {
                preprocessing::balanced_preprocess(&img, self.config.target_resolution)?
            }
            PreprocessingQuality::Quality => {
                preprocessing::quality_preprocess(&img, self.config.target_resolution)?
            }
        };

        // Extract visual features
        let tensor = self.feature_extractor.extract_features(&processed_img)?;
        
        // Analyze image metadata
        let metadata = self.analyze_image_metadata(&processed_img, format, file_size)?;

        let (width, height) = processed_img.dimensions();
        let channels = match processed_img {
            DynamicImage::ImageRgb8(_) => 3,
            DynamicImage::ImageRgba8(_) => 4,
            DynamicImage::ImageLuma8(_) => 1,
            DynamicImage::ImageLumaA8(_) => 2,
            _ => 3, // Default to RGB
        };

        Ok(ImageFeatures {
            tensor,
            width,
            height,
            channels,
            metadata,
        })
    }

    /// Analyze image metadata including objects, colors, etc.
    fn analyze_image_metadata(
        &self,
        img: &DynamicImage,
        format: String,
        file_size: Option<usize>
    ) -> Result<ImageMetadata> {
        // Extract color profile
        let color_profile = self.analyze_color_profile(img)?;
        
        // Detect objects if enabled
        let detected_objects = if self.config.enable_object_detection {
            self.detect_objects(img)?
        } else {
            Vec::new()
        };

        // Check for faces if enabled
        let has_faces = if self.config.enable_face_detection {
            self.detect_faces(img)?
        } else {
            false
        };

        Ok(ImageMetadata {
            format,
            file_size,
            has_faces,
            detected_objects,
            color_profile,
        })
    }

    /// Analyze color properties of the image
    fn analyze_color_profile(&self, img: &DynamicImage) -> Result<ColorProfile> {
        let rgb_img = img.to_rgb8();
        let pixels: Vec<_> = rgb_img.pixels().collect();
        
        // Calculate dominant colors (simplified)
        let mut color_counts = std::collections::HashMap::new();
        for pixel in &pixels {
            let color = [pixel[0], pixel[1], pixel[2]];
            *color_counts.entry(color).or_insert(0) += 1;
        }
        
        let mut dominant_colors: Vec<_> = color_counts.into_iter().collect();
        dominant_colors.sort_by(|a, b| b.1.cmp(&a.1));
        
        let dominant_colors: Vec<[u8; 3]> = dominant_colors
            .into_iter()
            .take(5)
            .map(|(color, _)| color)
            .collect();

        // Calculate brightness
        let brightness = pixels.iter()
            .map(|p| (p[0] as f32 + p[1] as f32 + p[2] as f32) / 3.0 / 255.0)
            .sum::<f32>() / pixels.len() as f32;

        // Calculate contrast and saturation (simplified)
        let contrast = self.calculate_contrast(&pixels);
        let saturation = self.calculate_saturation(&pixels);
        
        Ok(ColorProfile {
            dominant_colors,
            brightness,
            contrast,
            saturation,
        })
    }

    /// Calculate contrast using standard deviation of brightness
    fn calculate_contrast(&self, pixels: &[&image::Rgb<u8>]) -> f32 {
        let brightness_values: Vec<f32> = pixels.iter()
            .map(|p| (p[0] as f32 + p[1] as f32 + p[2] as f32) / 3.0 / 255.0)
            .collect();
        
        let mean = brightness_values.iter().sum::<f32>() / brightness_values.len() as f32;
        let variance = brightness_values.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f32>() / brightness_values.len() as f32;
        
        variance.sqrt() // Standard deviation as contrast measure
    }
    
    /// Calculate saturation as average color saturation
    fn calculate_saturation(&self, pixels: &[&image::Rgb<u8>]) -> f32 {
        let saturations: Vec<f32> = pixels.iter().map(|p| {
            let r = p[0] as f32 / 255.0;
            let g = p[1] as f32 / 255.0;
            let b = p[2] as f32 / 255.0;
            
            let max = r.max(g).max(b);
            let min = r.min(g).min(b);
            
            if max == 0.0 { 0.0 } else { (max - min) / max }
        }).collect();
        
        saturations.iter().sum::<f32>() / saturations.len() as f32
    }

    /// Detect objects in the image (disabled for now)
    fn detect_objects(&self, _img: &DynamicImage) -> Result<Vec<DetectedObject>> {
        // Object detection would require a separate ML model
        // For now, return empty vector
        Ok(Vec::new())
    }

    /// Detect faces in the image (disabled for now)
    fn detect_faces(&self, _img: &DynamicImage) -> Result<bool> {
        // Face detection would require a separate ML model
        // For now, return false
        Ok(false)
    }

    /// Detect image format from magic bytes
    fn detect_image_format(&self, data: &[u8]) -> Result<String> {
        if data.len() < 8 {
            return Ok("unknown".to_string());
        }

        let format = match &data[0..8] {
            [0xFF, 0xD8, 0xFF, ..] => "jpeg",
            [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A] => "png",
            [0x47, 0x49, 0x46, 0x38, 0x37, 0x61, ..] | [0x47, 0x49, 0x46, 0x38, 0x39, 0x61, ..] => "gif",
            [0x52, 0x49, 0x46, 0x46, _, _, _, _] if &data[8..12] == b"WEBP" => "webp",
            [0x42, 0x4D, ..] => "bmp",
            [0x49, 0x49, 0x2A, 0x00, ..] | [0x4D, 0x4D, 0x00, 0x2A, ..] => "tiff",
            _ => "unknown",
        };

        Ok(format.to_string())
    }

    /// Get supported image formats
    pub fn supported_formats(&self) -> &[String] {
        &self.config.supported_formats
    }
}

use nokhwa::utils::{CameraFormat, FrameFormat};
use nokhwa::Camera;
use std::sync::mpsc::{channel, Receiver, Sender};
use std::thread;

pub enum CameraCommand {
    StartCapture,
    StopCapture,
    GetFrame,
}

pub enum CameraResponse {
    CaptureStarted,
    CaptureStopped,
    Frame(CameraFrame),
    Error(String),
}

pub struct VisionManager {
    app_handle: AppHandle,
    is_capturing: Arc<Mutex<bool>>,
    command_tx: Sender<CameraCommand>,
    response_rx: Arc<Mutex<Receiver<CameraResponse>>>,
    _camera_thread: thread::JoinHandle<()>,
}

impl VisionManager {
    pub fn new(app_handle: AppHandle) -> Result<Self> {
        let (command_tx, command_rx) = channel();
        let (response_tx, response_rx) = channel();
        let response_rx = Arc::new(Mutex::new(response_rx));

        let camera_thread = thread::spawn(move || {
            if let Err(e) = camera_thread_main(command_rx, response_tx) {
                error!("Camera thread error: {}", e);
            }
        });

        Ok(Self {
            app_handle,
            is_capturing: Arc::new(Mutex::new(false)),
            command_tx,
            response_rx,
            _camera_thread: camera_thread,
        })
    }

    pub async fn initialize_camera(&self) -> Result<()> {
        // Initialization is now handled in the camera thread
        Ok(())
    }

    pub async fn start_capture(&self) -> Result<()> {
        self.command_tx.send(CameraCommand::StartCapture)?;
        match self.response_rx.lock().unwrap().recv()? {
            CameraResponse::CaptureStarted => {
                *self.is_capturing.lock().unwrap() = true;
                let _ = self.app_handle.emit_to(tauri::EventTarget::Any, "camera-capture-started", true);
                Ok(())
            }
            CameraResponse::Error(e) => Err(anyhow::anyhow!(e)),
            _ => Err(anyhow::anyhow!("Unexpected response from camera thread")),
        }
    }

    pub async fn stop_capture(&self) -> Result<()> {
        self.command_tx.send(CameraCommand::StopCapture)?;
        match self.response_rx.lock().unwrap().recv()? {
            CameraResponse::CaptureStopped => {
                *self.is_capturing.lock().unwrap() = false;
                let _ = self.app_handle.emit_to(tauri::EventTarget::Any, "camera-capture-stopped", true);
                Ok(())
            }
            CameraResponse::Error(e) => Err(anyhow::anyhow!(e)),
            _ => Err(anyhow::anyhow!("Unexpected response from camera thread")),
        }
    }

    pub async fn capture_frame(&self) -> Result<CameraFrame> {
        self.command_tx.send(CameraCommand::GetFrame)?;
        match self.response_rx.lock().unwrap().recv()? {
            CameraResponse::Frame(frame) => Ok(frame),
            CameraResponse::Error(e) => Err(anyhow::anyhow!(e)),
            _ => Err(anyhow::anyhow!("Unexpected response from camera thread")),
        }
    }

    pub fn is_capturing(&self) -> bool {
        *self.is_capturing.lock().unwrap()
    }
}

// Helper function to convert NV12 format to RGB
fn convert_nv12_to_rgb(frame: &nokhwa::Buffer) -> Result<image::ImageBuffer<image::Rgb<u8>, Vec<u8>>, nokhwa::NokhwaError> {
    let resolution = frame.resolution();
    let width = resolution.width_x;
    let height = resolution.height_y;
    let data = frame.buffer();
    
    // Debug buffer size issue
    info!("NV12 conversion attempt: {}x{}, buffer size: {} bytes", width, height, data.len());
    
    // NV12 format: Y plane followed by interleaved UV plane
    let y_size = (width * height) as usize;
    let standard_nv12_size = y_size + (y_size / 2); // Y + UV/2
    
    info!("Expected standard NV12 size: {} bytes, actual: {} bytes", standard_nv12_size, data.len());
    
    // Check for different possible layouts
    let uv_size = if data.len() == y_size * 2 {
        // This might be YUYV format mislabeled as NV12, or NV12 with full UV plane
        info!("Buffer size suggests YUYV format or padded NV12");
        y_size // Use full Y plane size for UV
    } else if data.len() >= standard_nv12_size {
        // Standard NV12 or with padding
        info!("Using standard NV12 layout");
        y_size / 2
    } else {
        return Err(nokhwa::NokhwaError::ProcessFrameError {
            src: nokhwa::utils::FrameFormat::NV12,
            destination: "RGB".to_string(),
            error: format!("Buffer too small: {} bytes for {}x{} (need at least {})", 
                           data.len(), width, height, standard_nv12_size),
        });
    };
    
    // Ensure we don't read beyond buffer
    let safe_uv_size = std::cmp::min(uv_size, data.len() - y_size);
    info!("Using Y plane: {} bytes, UV plane: {} bytes", y_size, safe_uv_size);
    
    let y_plane = &data[0..y_size];
    let uv_plane = &data[y_size..y_size + safe_uv_size];
    
    let mut rgb_data = vec![0u8; (width * height * 3) as usize];
    
    for y in 0..height {
        for x in 0..width {
            let y_index = (y * width + x) as usize;
            let uv_index = (((y / 2) * (width / 2) + (x / 2)) * 2) as usize;
            
            let y_val = y_plane[y_index] as f32;
            let u_val = if uv_index < uv_plane.len() { uv_plane[uv_index] as f32 - 128.0 } else { 0.0 };
            let v_val = if uv_index + 1 < uv_plane.len() { uv_plane[uv_index + 1] as f32 - 128.0 } else { 0.0 };
            
            // YUV to RGB conversion
            let r = (y_val + 1.402 * v_val).clamp(0.0, 255.0) as u8;
            let g = (y_val - 0.344 * u_val - 0.714 * v_val).clamp(0.0, 255.0) as u8;
            let b = (y_val + 1.772 * u_val).clamp(0.0, 255.0) as u8;
            
            let rgb_index = y_index * 3;
            if rgb_index + 2 < rgb_data.len() {
                rgb_data[rgb_index] = r;
                rgb_data[rgb_index + 1] = g;
                rgb_data[rgb_index + 2] = b;
            }
        }
    }
    
    match image::ImageBuffer::from_raw(width, height, rgb_data) {
        Some(img) => Ok(img),
        None => Err(nokhwa::NokhwaError::ProcessFrameError {
            src: nokhwa::utils::FrameFormat::NV12,
            destination: "RGB".to_string(),
            error: "Failed to create RGB image buffer from converted data".to_string(),
        })
    }
}

// Helper function to convert various frame formats to RGB
fn convert_frame_to_rgb(frame: &nokhwa::Buffer) -> Result<image::ImageBuffer<image::Rgb<u8>, Vec<u8>>, nokhwa::NokhwaError> {
    use nokhwa::pixel_format::*;
    
    match frame.source_frame_format() {
        nokhwa::utils::FrameFormat::MJPEG => {
            // MJPEG should decode normally, this is a fallback
            frame.decode_image::<RgbFormat>()
        }
        nokhwa::utils::FrameFormat::YUYV => {
            // Try YUYV format
            frame.decode_image::<YuyvFormat>()
                .and_then(|img| {
                    // Convert YUYV to RGB
                    Ok(img) // YuyvFormat should already be RGB-compatible
                })
        }
        nokhwa::utils::FrameFormat::NV12 => {
            // NV12 format handling - implement basic conversion
            info!("Converting NV12 to RGB");
            convert_nv12_to_rgb(frame)
        }
        _ => {
            // For other formats, try the default RGB decode
            frame.decode_image::<RgbFormat>()
        }
    }
}

fn camera_thread_main(
    command_rx: Receiver<CameraCommand>,
    response_tx: Sender<CameraResponse>,
) -> Result<()> {
    let cameras = nokhwa::query(nokhwa::utils::ApiBackend::Auto)?;
    let camera_info = cameras.first().ok_or_else(|| anyhow::anyhow!("No cameras found"))?;

    // Try different formats until one works
    let formats_to_try = vec![
        FrameFormat::MJPEG,
        FrameFormat::YUYV,
        FrameFormat::NV12,
        FrameFormat::RAWRGB
    ];
    
    let mut camera = None;
    for format in formats_to_try {
        match Camera::new(
            camera_info.index().clone(),
            nokhwa::utils::RequestedFormat::new::<nokhwa::pixel_format::RgbFormat>(
                nokhwa::utils::RequestedFormatType::Exact(CameraFormat::new_from(640, 480, format, 30))
            ),
        ) {
            Ok(cam) => {
                info!("Successfully initialized camera with format: {:?}", format);
                camera = Some(cam);
                break;
            }
            Err(e) => {
                info!("Failed to initialize camera with format {:?}: {}", format, e);
                continue;
            }
        }
    }
    
    let mut camera = camera.ok_or_else(|| anyhow::anyhow!("Failed to initialize camera with any supported format"))?;

    let mut is_capturing = false;
    let mut conversion_failure_count = 0;
    const MAX_CONVERSION_FAILURES: u32 = 3;

    loop {
        match command_rx.try_recv() {
            Ok(CameraCommand::StartCapture) => {
                if !is_capturing {
                    if let Err(e) = camera.open_stream() {
                        let _ = response_tx.send(CameraResponse::Error(e.to_string()));
                        continue;
                    }
                    is_capturing = true;
                }
                let _ = response_tx.send(CameraResponse::CaptureStarted);
            }
            Ok(CameraCommand::StopCapture) => {
                if is_capturing {
                    camera.stop_stream()?;
                    is_capturing = false;
                }
                let _ = response_tx.send(CameraResponse::CaptureStopped);
            }
            Ok(CameraCommand::GetFrame) => {
                if is_capturing && conversion_failure_count < MAX_CONVERSION_FAILURES {
                    match camera.frame() {
                        Ok(frame) => {
                            // Try to decode directly first
                            let decoded_result = frame.decode_image::<nokhwa::pixel_format::RgbFormat>()
                                .or_else(|initial_error| {
                                    // Only log on first few failures to avoid spam
                                    if conversion_failure_count == 0 {
                                        info!("Direct RGB decode failed for format {:?}: {}. Trying format-specific conversion.", 
                                            frame.source_frame_format(), initial_error);
                                    }
                                    
                                    // Try format-specific conversion
                                    convert_frame_to_rgb(&frame)
                                });
                            
                            match decoded_result {
                                Ok(decoded) => {
                                    // Reset failure count on success
                                    conversion_failure_count = 0;
                                    let _ = response_tx.send(CameraResponse::Frame(CameraFrame {
                                        width: decoded.width(),
                                        height: decoded.height(),
                                        data: decoded.to_vec(),
                                        timestamp: std::time::SystemTime::now()
                                            .duration_since(std::time::UNIX_EPOCH)
                                            .unwrap()
                                            .as_millis() as u64,
                                    }));
                                }
                                Err(_e) => {
                                    conversion_failure_count += 1;
                                    
                                    if conversion_failure_count >= MAX_CONVERSION_FAILURES {
                                        // Send final error message and stop trying
                                        let error_msg = format!("Camera format {} not supported after {} attempts. Stopping camera capture to prevent log spam.", 
                                            frame.source_frame_format(), MAX_CONVERSION_FAILURES);
                                        let _ = response_tx.send(CameraResponse::Error(error_msg));
                                        
                                        // Stop trying to capture frames
                                        camera.stop_stream().ok();
                                        is_capturing = false;
                                        info!("Camera capture disabled due to repeated conversion failures");
                                    }
                                }
                            }
                        }
                        Err(e) => {
                            let _ = response_tx.send(CameraResponse::Error(e.to_string()));
                        }
                    }
                } else if conversion_failure_count >= MAX_CONVERSION_FAILURES {
                    // Don't attempt any more frame captures
                    let _ = response_tx.send(CameraResponse::Error("Camera disabled due to format conversion failures".to_string()));
                } else {
                    let _ = response_tx.send(CameraResponse::Error("Camera not capturing".to_string()));
                }
            }
            Err(_) => {
                // No command, do nothing
            }
        }
        thread::sleep(std::time::Duration::from_millis(10));
    }
}

// Tauri command to get camera frame as base64
pub async fn get_camera_frame_base64(vision: &VisionManager) -> Result<String> {
    let frame = vision.capture_frame().await?;
    
    // Create a proper image from RGB data
    use base64::{engine::general_purpose, Engine as _};
    
    // Convert RGB to PNG for better browser compatibility
    let png_data = rgb_to_png(&frame.data, frame.width, frame.height)?;
    let base64_data = general_purpose::STANDARD.encode(&png_data);
    
    Ok(format!("data:image/png;base64,{}", base64_data))
}

// Helper function to convert RGB to PNG
fn rgb_to_png(rgb_data: &[u8], width: u32, height: u32) -> Result<Vec<u8>> {
    use std::io::Cursor;
    
    // Create PNG encoder
    let mut png_data = Vec::new();
    {
        let mut encoder = png::Encoder::new(Cursor::new(&mut png_data), width, height);
        encoder.set_color(png::ColorType::Rgb);
        encoder.set_depth(png::BitDepth::Eight);
        
        let mut writer = encoder.write_header()?;
        writer.write_image_data(rgb_data)?;
    }
    
    Ok(png_data)
}