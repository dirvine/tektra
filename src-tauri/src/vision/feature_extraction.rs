use anyhow::Result;
use image::DynamicImage;
use tracing::{info, warn};
use crate::vision::{VisionConfig, PreprocessingQuality};

/// Feature extractor for multimodal AI models
pub struct FeatureExtractor {
    config: VisionConfig,
    feature_dim: usize,
}

impl FeatureExtractor {
    pub fn new(config: &VisionConfig) -> Result<Self> {
        info!("Initializing feature extractor for target resolution: {:?}", config.target_resolution);
        
        // Calculate feature dimension based on target resolution
        let (width, height) = config.target_resolution;
        let feature_dim = (width * height * 3) as usize; // RGB channels
        
        Ok(Self {
            config: config.clone(),
            feature_dim,
        })
    }

    /// Extract features from a processed image for AI model input
    pub fn extract_features(&self, img: &DynamicImage) -> Result<Vec<f32>> {
        info!("Extracting features from {}x{} image", img.width(), img.height());
        
        // Ensure image matches target resolution
        let (target_width, target_height) = self.config.target_resolution;
        if img.width() != target_width || img.height() != target_height {
            return Err(anyhow::anyhow!(
                "Image dimensions {}x{} don't match target {}x{}", 
                img.width(), img.height(), target_width, target_height
            ));
        }

        // Convert to RGB if needed
        let rgb_img = img.to_rgb8();
        
        // Extract features based on preprocessing quality
        match self.config.preprocessing_quality {
            PreprocessingQuality::Fast => self.extract_fast_features(&rgb_img),
            PreprocessingQuality::Balanced => self.extract_balanced_features(&rgb_img),
            PreprocessingQuality::Quality => self.extract_quality_features(&rgb_img),
        }
    }

    /// Fast feature extraction - simple pixel normalization
    fn extract_fast_features(&self, img: &image::RgbImage) -> Result<Vec<f32>> {
        info!("Extracting fast features");
        
        let features: Vec<f32> = img.pixels()
            .flat_map(|pixel| {
                // Simple normalization to [0, 1] range
                pixel.0.iter().map(|&channel| channel as f32 / 255.0)
            })
            .collect();
        
        Ok(features)
    }

    /// Balanced feature extraction - normalized with basic statistics
    fn extract_balanced_features(&self, img: &image::RgbImage) -> Result<Vec<f32>> {
        info!("Extracting balanced features");
        
        // Calculate basic image statistics for normalization
        let pixels: Vec<_> = img.pixels().collect();
        let mut channel_stats = vec![[0.0f32; 3]; 2]; // [means, stds]
        
        // Calculate means
        for pixel in &pixels {
            for (i, &channel) in pixel.0.iter().enumerate() {
                channel_stats[0][i] += channel as f32;
            }
        }
        
        let pixel_count = pixels.len() as f32;
        for i in 0..3 {
            channel_stats[0][i] /= pixel_count;
        }
        
        // Calculate standard deviations
        for pixel in &pixels {
            for (i, &channel) in pixel.0.iter().enumerate() {
                let diff = channel as f32 - channel_stats[0][i];
                channel_stats[1][i] += diff * diff;
            }
        }
        
        for i in 0..3 {
            channel_stats[1][i] = (channel_stats[1][i] / pixel_count).sqrt();
            // Avoid division by zero
            if channel_stats[1][i] < 1.0 {
                channel_stats[1][i] = 1.0;
            }
        }
        
        // Normalize using calculated statistics
        let features: Vec<f32> = img.pixels()
            .flat_map(|pixel| {
                pixel.0.iter().enumerate().map(|(i, &channel)| {
                    (channel as f32 - channel_stats[0][i]) / channel_stats[1][i]
                })
            })
            .collect();
        
        Ok(features)
    }

    /// Quality feature extraction - enhanced normalization with ImageNet stats
    fn extract_quality_features(&self, img: &image::RgbImage) -> Result<Vec<f32>> {
        info!("Extracting quality features with ImageNet normalization");
        
        // ImageNet statistics (commonly used in vision models)
        let imagenet_mean = [0.485, 0.456, 0.406];
        let imagenet_std = [0.229, 0.224, 0.225];
        
        let features: Vec<f32> = img.pixels()
            .flat_map(|pixel| {
                pixel.0.iter().enumerate().map(|(i, &channel)| {
                    // Normalize to [0, 1] then apply ImageNet normalization
                    let normalized = channel as f32 / 255.0;
                    (normalized - imagenet_mean[i]) / imagenet_std[i]
                })
            })
            .collect();
        
        Ok(features)
    }

    /// Extract spatial features (edge detection, texture analysis)
    pub fn extract_spatial_features(&self, img: &DynamicImage) -> Result<Vec<f32>> {
        info!("Extracting spatial features");
        
        let gray_img = img.to_luma8();
        let (width, height) = gray_img.dimensions();
        
        // Simple edge detection using Sobel operators
        let mut edge_features = Vec::new();
        
        for y in 1..height-1 {
            for x in 1..width-1 {
                // Sobel X kernel
                let gx = 
                    -1.0 * gray_img.get_pixel(x-1, y-1)[0] as f32 +
                     1.0 * gray_img.get_pixel(x+1, y-1)[0] as f32 +
                    -2.0 * gray_img.get_pixel(x-1, y)[0] as f32 +
                     2.0 * gray_img.get_pixel(x+1, y)[0] as f32 +
                    -1.0 * gray_img.get_pixel(x-1, y+1)[0] as f32 +
                     1.0 * gray_img.get_pixel(x+1, y+1)[0] as f32;
                
                // Sobel Y kernel
                let gy = 
                    -1.0 * gray_img.get_pixel(x-1, y-1)[0] as f32 +
                    -2.0 * gray_img.get_pixel(x, y-1)[0] as f32 +
                    -1.0 * gray_img.get_pixel(x+1, y-1)[0] as f32 +
                     1.0 * gray_img.get_pixel(x-1, y+1)[0] as f32 +
                     2.0 * gray_img.get_pixel(x, y+1)[0] as f32 +
                     1.0 * gray_img.get_pixel(x+1, y+1)[0] as f32;
                
                // Edge magnitude
                let magnitude = (gx * gx + gy * gy).sqrt() / 255.0;
                edge_features.push(magnitude);
            }
        }
        
        // Downsample edge features to manageable size
        let downsample_factor = ((edge_features.len() as f32 / 1000.0).sqrt().ceil() as usize).max(1);
        let downsampled: Vec<f32> = edge_features.into_iter()
            .step_by(downsample_factor)
            .collect();
        
        info!("Extracted {} spatial features", downsampled.len());
        Ok(downsampled)
    }

    /// Extract color histogram features
    pub fn extract_color_histogram(&self, img: &DynamicImage) -> Result<Vec<f32>> {
        info!("Extracting color histogram features");
        
        let rgb_img = img.to_rgb8();
        let mut histograms = vec![vec![0u32; 256]; 3]; // RGB histograms
        
        // Build histograms
        for pixel in rgb_img.pixels() {
            for (channel, &value) in pixel.0.iter().enumerate() {
                histograms[channel][value as usize] += 1;
            }
        }
        
        // Normalize histograms
        let total_pixels = rgb_img.pixels().len() as f32;
        let normalized_histograms: Vec<f32> = histograms.into_iter()
            .flat_map(|hist| {
                hist.into_iter().map(|count| count as f32 / total_pixels)
            })
            .collect();
        
        info!("Extracted {} histogram features", normalized_histograms.len());
        Ok(normalized_histograms)
    }

    /// Combine multiple feature types into a single feature vector
    pub fn extract_combined_features(&self, img: &DynamicImage) -> Result<Vec<f32>> {
        info!("Extracting combined feature vector");
        
        let mut combined_features = Vec::new();
        
        // Primary features (pixel values)
        let primary_features = self.extract_features(img)?;
        combined_features.extend(primary_features);
        
        // Spatial features (edges, texture)
        match self.extract_spatial_features(img) {
            Ok(spatial_features) => {
                combined_features.extend(spatial_features);
            }
            Err(e) => {
                warn!("Failed to extract spatial features: {}", e);
            }
        }
        
        // Color histogram features
        match self.extract_color_histogram(img) {
            Ok(color_features) => {
                combined_features.extend(color_features);
            }
            Err(e) => {
                warn!("Failed to extract color features: {}", e);
            }
        }
        
        info!("Combined feature vector size: {}", combined_features.len());
        Ok(combined_features)
    }

    /// Get the expected feature dimension for the primary features
    pub fn feature_dim(&self) -> usize {
        self.feature_dim
    }

    /// Validate that extracted features have the expected dimensions
    pub fn validate_features(&self, features: &[f32]) -> Result<()> {
        if features.len() != self.feature_dim {
            return Err(anyhow::anyhow!(
                "Feature dimension mismatch: expected {}, got {}", 
                self.feature_dim, features.len()
            ));
        }
        
        // Check for NaN or infinite values
        for (i, &value) in features.iter().enumerate() {
            if !value.is_finite() {
                return Err(anyhow::anyhow!(
                    "Invalid feature value at index {}: {}", i, value
                ));
            }
        }
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::RgbImage;

    #[test]
    fn test_feature_extractor_creation() {
        let config = VisionConfig::default();
        let extractor = FeatureExtractor::new(&config);
        assert!(extractor.is_ok());
        
        let extractor = extractor.unwrap();
        assert_eq!(extractor.feature_dim(), 224 * 224 * 3);
    }

    #[test]
    fn test_fast_feature_extraction() {
        let config = VisionConfig {
            target_resolution: (8, 8),
            preprocessing_quality: PreprocessingQuality::Fast,
            ..Default::default()
        };
        
        let extractor = FeatureExtractor::new(&config).unwrap();
        let test_img = DynamicImage::ImageRgb8(RgbImage::new(8, 8));
        
        let features = extractor.extract_features(&test_img);
        assert!(features.is_ok());
        
        let features = features.unwrap();
        assert_eq!(features.len(), 8 * 8 * 3);
        
        // Features should be normalized to [0, 1]
        for &feature in &features {
            assert!(feature >= 0.0 && feature <= 1.0);
        }
    }

    #[test]
    fn test_feature_validation() {
        let config = VisionConfig {
            target_resolution: (8, 8),
            ..Default::default()
        };
        
        let extractor = FeatureExtractor::new(&config).unwrap();
        
        // Valid features
        let valid_features = vec![0.5; 8 * 8 * 3];
        assert!(extractor.validate_features(&valid_features).is_ok());
        
        // Invalid dimension
        let invalid_features = vec![0.5; 10];
        assert!(extractor.validate_features(&invalid_features).is_err());
        
        // Invalid values (NaN)
        let mut nan_features = vec![0.5; 8 * 8 * 3];
        nan_features[0] = f32::NAN;
        assert!(extractor.validate_features(&nan_features).is_err());
    }

    #[test]
    fn test_color_histogram() {
        let config = VisionConfig {
            target_resolution: (8, 8),
            ..Default::default()
        };
        
        let extractor = FeatureExtractor::new(&config).unwrap();
        let test_img = DynamicImage::ImageRgb8(RgbImage::new(8, 8));
        
        let histogram = extractor.extract_color_histogram(&test_img);
        assert!(histogram.is_ok());
        
        let histogram = histogram.unwrap();
        assert_eq!(histogram.len(), 3 * 256); // RGB * 256 bins
        
        // Histogram should be normalized (sum to 1 for each channel)
        for channel in 0..3 {
            let channel_sum: f32 = histogram[channel * 256..(channel + 1) * 256].iter().sum();
            assert!((channel_sum - 1.0).abs() < 0.001);
        }
    }
}