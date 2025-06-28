use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};
use tauri::{AppHandle, Manager};
use tracing::info;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CameraFrame {
    pub width: u32,
    pub height: u32,
    pub data: Vec<u8>, // RGB data
    pub timestamp: u64,
}

pub struct VisionManager {
    app_handle: AppHandle,
    is_capturing: Arc<Mutex<bool>>,
    current_frame: Arc<Mutex<Option<CameraFrame>>>,
    // In test mode, we'll simulate camera capture
    test_mode: bool,
}

impl VisionManager {
    pub fn new(app_handle: AppHandle) -> Self {
        Self {
            app_handle,
            is_capturing: Arc::new(Mutex::new(false)),
            current_frame: Arc::new(Mutex::new(None)),
            test_mode: true, // Always in test mode for now
        }
    }

    pub async fn initialize_camera(&self) -> Result<()> {
        info!("Initializing camera in test mode...");
        
        if self.test_mode {
            info!("Camera test mode active - will simulate camera frames");
            let _ = self.app_handle.emit_all("camera-ready", true);
            Ok(())
        } else {
            Err(anyhow::anyhow!("Real camera support not yet implemented"))
        }
    }

    pub async fn start_capture(&self) -> Result<()> {
        if *self.is_capturing.lock().unwrap() {
            return Err(anyhow::anyhow!("Already capturing"));
        }

        *self.is_capturing.lock().unwrap() = true;
        info!("Camera capture started (test mode)");
        let _ = self.app_handle.emit_all("camera-capture-started", true);
        
        // Start simulated capture loop
        self.start_test_capture_loop();
        
        Ok(())
    }

    pub async fn stop_capture(&self) -> Result<()> {
        if !*self.is_capturing.lock().unwrap() {
            return Err(anyhow::anyhow!("Not capturing"));
        }

        *self.is_capturing.lock().unwrap() = false;
        info!("Camera capture stopped");
        let _ = self.app_handle.emit_all("camera-capture-stopped", true);

        Ok(())
    }

    pub async fn capture_frame(&self) -> Result<CameraFrame> {
        if !*self.is_capturing.lock().unwrap() {
            return Err(anyhow::anyhow!("Camera not capturing"));
        }

        // Get the current frame from cache
        if let Some(frame) = self.current_frame.lock().unwrap().clone() {
            Ok(frame)
        } else {
            Err(anyhow::anyhow!("No frame available"))
        }
    }
    
    fn start_test_capture_loop(&self) {
        let is_capturing = Arc::clone(&self.is_capturing);
        let current_frame = Arc::clone(&self.current_frame);
        let app_handle = self.app_handle.clone();
        
        std::thread::spawn(move || {
            let mut frame_count = 0u32;
            
            while *is_capturing.lock().unwrap() {
                // Generate a test pattern frame
                let width = 640u32;
                let height = 480u32;
                let mut data = vec![0u8; (width * height * 3) as usize];
                
                // Create a simple gradient pattern that changes over time
                for y in 0..height {
                    for x in 0..width {
                        let idx = ((y * width + x) * 3) as usize;
                        // Red channel: horizontal gradient
                        data[idx] = ((x as f32 / width as f32) * 255.0) as u8;
                        // Green channel: vertical gradient  
                        data[idx + 1] = ((y as f32 / height as f32) * 255.0) as u8;
                        // Blue channel: animated based on frame count
                        data[idx + 2] = ((frame_count % 255) as f32) as u8;
                    }
                }
                
                let frame = CameraFrame {
                    width,
                    height,
                    data,
                    timestamp: std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap()
                        .as_millis() as u64,
                };
                
                // Store the frame
                *current_frame.lock().unwrap() = Some(frame);
                
                // Emit frame event (throttled)
                let _ = app_handle.emit_all("camera-frame", true);
                
                frame_count = frame_count.wrapping_add(1);
                
                // Small delay to simulate ~30 FPS
                std::thread::sleep(std::time::Duration::from_millis(33));
            }
        });
    }

    pub fn is_capturing(&self) -> bool {
        *self.is_capturing.lock().unwrap()
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