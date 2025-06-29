use anyhow::Result;
use image::{DynamicImage, ImageBuffer, Rgba, GenericImageView};
use tracing::info;

/// Fast preprocessing for real-time applications
pub fn fast_preprocess(img: &DynamicImage, target_resolution: (u32, u32)) -> Result<DynamicImage> {
    info!("Applying fast preprocessing to {}x{} -> {}x{}", 
          img.width(), img.height(), target_resolution.0, target_resolution.1);
    
    // Simple resize with nearest neighbor for speed
    let resized = img.resize_exact(
        target_resolution.0,
        target_resolution.1,
        image::imageops::FilterType::Nearest
    );
    
    // Convert to RGB if needed
    Ok(DynamicImage::ImageRgb8(resized.to_rgb8()))
}

/// Balanced preprocessing with good quality/performance tradeoff
pub fn balanced_preprocess(img: &DynamicImage, target_resolution: (u32, u32)) -> Result<DynamicImage> {
    info!("Applying balanced preprocessing to {}x{} -> {}x{}", 
          img.width(), img.height(), target_resolution.0, target_resolution.1);
    
    let mut processed = img.clone();
    
    // Step 1: Resize with Lanczos3 for good quality
    processed = processed.resize_exact(
        target_resolution.0,
        target_resolution.1,
        image::imageops::FilterType::Lanczos3
    );
    
    // Step 2: Basic color normalization
    processed = normalize_colors(&processed)?;
    
    // Step 3: Convert to RGB
    Ok(DynamicImage::ImageRgb8(processed.to_rgb8()))
}

/// High-quality preprocessing for best results
pub fn quality_preprocess(img: &DynamicImage, target_resolution: (u32, u32)) -> Result<DynamicImage> {
    info!("Applying quality preprocessing to {}x{} -> {}x{}", 
          img.width(), img.height(), target_resolution.0, target_resolution.1);
    
    let mut processed = img.clone();
    
    // Step 1: Noise reduction (simplified)
    processed = reduce_noise(&processed)?;
    
    // Step 2: Smart resize maintaining aspect ratio
    processed = smart_resize(&processed, target_resolution)?;
    
    // Step 3: Enhanced color normalization
    processed = normalize_colors_enhanced(&processed)?;
    
    // Step 4: Sharpening (subtle)
    processed = apply_sharpening(&processed)?;
    
    // Step 5: Convert to RGB
    Ok(DynamicImage::ImageRgb8(processed.to_rgb8()))
}

/// Smart resize that maintains aspect ratio and adds padding if needed
fn smart_resize(img: &DynamicImage, target_resolution: (u32, u32)) -> Result<DynamicImage> {
    let (target_width, target_height) = target_resolution;
    let (orig_width, orig_height) = img.dimensions();
    
    // Calculate scaling factor to fit within target resolution
    let scale_x = target_width as f32 / orig_width as f32;
    let scale_y = target_height as f32 / orig_height as f32;
    let scale = scale_x.min(scale_y);
    
    // Calculate new dimensions
    let new_width = (orig_width as f32 * scale) as u32;
    let new_height = (orig_height as f32 * scale) as u32;
    
    // Resize the image
    let resized = img.resize(new_width, new_height, image::imageops::FilterType::Lanczos3);
    
    // If the resized image doesn't match target exactly, add padding
    if new_width != target_width || new_height != target_height {
        let mut padded = ImageBuffer::from_pixel(target_width, target_height, Rgba([0u8, 0u8, 0u8, 255u8]));
        
        // Calculate center position
        let x_offset = (target_width - new_width) / 2;
        let y_offset = (target_height - new_height) / 2;
        
        // Copy resized image to center of padded image
        let resized_rgba = resized.to_rgba8();
        for y in 0..new_height {
            for x in 0..new_width {
                let pixel = resized_rgba.get_pixel(x, y);
                padded.put_pixel(x + x_offset, y + y_offset, *pixel);
            }
        }
        
        Ok(DynamicImage::ImageRgba8(padded))
    } else {
        Ok(resized)
    }
}

/// Basic color normalization
fn normalize_colors(img: &DynamicImage) -> Result<DynamicImage> {
    let rgb_img = img.to_rgb8();
    let mut normalized = rgb_img.clone();
    
    // Simple contrast enhancement
    for pixel in normalized.pixels_mut() {
        for channel in pixel.0.iter_mut() {
            // Apply basic gamma correction
            let normalized_val = (*channel as f32 / 255.0).powf(0.9);
            *channel = (normalized_val * 255.0) as u8;
        }
    }
    
    Ok(DynamicImage::ImageRgb8(normalized))
}

/// Enhanced color normalization with histogram equalization
fn normalize_colors_enhanced(img: &DynamicImage) -> Result<DynamicImage> {
    let rgb_img = img.to_rgb8();
    let mut enhanced = rgb_img.clone();
    
    // Calculate histogram for each channel
    for channel_idx in 0..3 {
        let mut histogram = [0u32; 256];
        
        // Build histogram
        for pixel in enhanced.pixels() {
            histogram[pixel.0[channel_idx] as usize] += 1;
        }
        
        // Calculate cumulative distribution
        let total_pixels = enhanced.width() * enhanced.height();
        let mut cdf = [0f32; 256];
        let mut cumulative = 0u32;
        
        for (i, &count) in histogram.iter().enumerate() {
            cumulative += count;
            cdf[i] = cumulative as f32 / total_pixels as f32;
        }
        
        // Apply histogram equalization
        for pixel in enhanced.pixels_mut() {
            let old_val = pixel.0[channel_idx] as usize;
            let new_val = (cdf[old_val] * 255.0) as u8;
            pixel.0[channel_idx] = new_val;
        }
    }
    
    Ok(DynamicImage::ImageRgb8(enhanced))
}

/// Simple noise reduction using a 3x3 gaussian blur
fn reduce_noise(img: &DynamicImage) -> Result<DynamicImage> {
    // Apply a subtle gaussian blur to reduce noise
    let blurred = img.blur(0.5);
    Ok(blurred)
}

/// Apply subtle sharpening to enhance edges
fn apply_sharpening(img: &DynamicImage) -> Result<DynamicImage> {
    let rgb_img = img.to_rgb8();
    let mut sharpened = rgb_img.clone();
    let (width, height) = rgb_img.dimensions();
    
    // Simple sharpening kernel
    let kernel = [
        [0.0, -0.1, 0.0],
        [-0.1, 1.4, -0.1],
        [0.0, -0.1, 0.0],
    ];
    
    // Apply sharpening (only to interior pixels to avoid boundary issues)
    for y in 1..height-1 {
        for x in 1..width-1 {
            let mut new_pixel = [0f32; 3];
            
            // Apply kernel
            for ky in 0..3 {
                for kx in 0..3 {
                    let px = (x as i32 + kx as i32 - 1) as u32;
                    let py = (y as i32 + ky as i32 - 1) as u32;
                    let pixel = rgb_img.get_pixel(px, py);
                    
                    for c in 0..3 {
                        new_pixel[c] += pixel.0[c] as f32 * kernel[ky][kx];
                    }
                }
            }
            
            // Clamp and set pixel
            let pixel = sharpened.get_pixel_mut(x, y);
            for c in 0..3 {
                pixel.0[c] = new_pixel[c].max(0.0).min(255.0) as u8;
            }
        }
    }
    
    Ok(DynamicImage::ImageRgb8(sharpened))
}

/// Utility function to convert image to normalized tensor format
pub fn image_to_tensor(img: &DynamicImage, normalize_range: (f32, f32)) -> Vec<f32> {
    let rgb_img = img.to_rgb8();
    let (min_val, max_val) = normalize_range;
    let range = max_val - min_val;
    
    rgb_img.pixels()
        .flat_map(|pixel| {
            pixel.0.iter().map(|&channel| {
                let normalized = channel as f32 / 255.0;
                min_val + normalized * range
            })
        })
        .collect()
}

/// Utility function to apply ImageNet-style normalization
pub fn imagenet_normalize(tensor: &mut [f32]) {
    // ImageNet mean and std for RGB channels
    let mean = [0.485, 0.456, 0.406];
    let std = [0.229, 0.224, 0.225];
    
    let pixels = tensor.len() / 3;
    for pixel_idx in 0..pixels {
        for channel in 0..3 {
            let idx = pixel_idx * 3 + channel;
            tensor[idx] = (tensor[idx] - mean[channel]) / std[channel];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::RgbImage;
    
    #[test]
    fn test_fast_preprocess() {
        let test_img = DynamicImage::ImageRgb8(RgbImage::new(100, 100));
        let result = fast_preprocess(&test_img, (224, 224));
        assert!(result.is_ok());
        
        let processed = result.unwrap();
        assert_eq!(processed.dimensions(), (224, 224));
    }
    
    #[test]
    fn test_image_to_tensor() {
        let test_img = DynamicImage::ImageRgb8(RgbImage::new(2, 2));
        let tensor = image_to_tensor(&test_img, (-1.0, 1.0));
        
        // 2x2 RGB image should produce 12 values
        assert_eq!(tensor.len(), 12);
        
        // All values should be in [-1, 1] range
        for &val in &tensor {
            assert!(val >= -1.0 && val <= 1.0);
        }
    }
    
    #[test]
    fn test_smart_resize() {
        let test_img = DynamicImage::ImageRgb8(RgbImage::new(100, 50));
        let result = smart_resize(&test_img, (224, 224));
        assert!(result.is_ok());
        
        let resized = result.unwrap();
        assert_eq!(resized.dimensions(), (224, 224));
    }
}