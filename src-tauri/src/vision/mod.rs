use anyhow::Result;
use image::{DynamicImage, ImageFormat, ImageBuffer, Rgb};
use std::sync::Arc;
use tokio::sync::Mutex;

pub struct VisionManager {
    current_frame: Arc<Mutex<Option<DynamicImage>>>,
    is_camera_available: bool,
}

impl VisionManager {
    pub fn new() -> Result<Self> {
        tracing::info!("Initializing vision manager (simulated mode)");
        
        // For now, we'll always use simulated camera to avoid thread safety issues
        // In a production implementation, you would use a thread-safe camera abstraction
        let is_camera_available = false;
        
        Ok(Self {
            current_frame: Arc::new(Mutex::new(None)),
            is_camera_available,
        })
    }
    
    pub async fn capture_frame(&self) -> Result<Vec<u8>> {
        // Always use simulated camera for now
        self.capture_simulated_frame().await
    }
    
    async fn capture_simulated_frame(&self) -> Result<Vec<u8>> {
        tracing::info!("Capturing simulated camera frame");
        
        // Create a synthetic test image with some pattern
        let width = 640u32;
        let height = 480u32;
        
        let mut img_buffer = ImageBuffer::new(width, height);
        
        // Create a simple gradient pattern with timestamp-based variation
        let time_factor = (std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs() % 60) as f32 / 60.0;
        
        for (x, y, pixel) in img_buffer.enumerate_pixels_mut() {
            let r = ((x as f32 / width as f32) * 255.0 * time_factor) as u8;
            let g = ((y as f32 / height as f32) * 255.0) as u8;
            let b = (((x + y) as f32 / (width + height) as f32) * 255.0 * (1.0 - time_factor)) as u8;
            *pixel = Rgb([r, g, b]);
        }
        
        // Add some visual indicators
        // Create a timestamp-based moving dot
        let dot_x = ((time_factor * width as f32) as u32).min(width - 10);
        let dot_y = height / 2;
        
        // Draw a small white square as a moving indicator
        for dx in 0..10 {
            for dy in 0..10 {
                if dot_x + dx < width && dot_y + dy < height {
                    let pixel = img_buffer.get_pixel_mut(dot_x + dx, dot_y + dy);
                    *pixel = Rgb([255, 255, 255]);
                }
            }
        }
        
        let dynamic_image = DynamicImage::ImageRgb8(img_buffer);
        *self.current_frame.lock().await = Some(dynamic_image.clone());
        
        // Encode as JPEG
        let mut buffer = Vec::new();
        dynamic_image.write_to(&mut std::io::Cursor::new(&mut buffer), ImageFormat::Jpeg)?;
        
        tracing::info!("Generated simulated frame: {}x{}, {} bytes", width, height, buffer.len());
        Ok(buffer)
    }
    
    pub async fn get_current_frame(&self) -> Option<DynamicImage> {
        self.current_frame.lock().await.clone()
    }
    
    pub async fn get_camera_info(&self) -> Result<String> {
        if self.is_camera_available {
            Ok("Real camera available".to_string())
        } else {
            Ok("Using simulated camera (640x480 RGB with animated pattern)".to_string())
        }
    }
    
    pub fn is_camera_available(&self) -> bool {
        self.is_camera_available
    }
}