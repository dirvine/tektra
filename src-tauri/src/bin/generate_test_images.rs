use image::{ImageBuffer, Rgb, RgbImage};
use imageproc::drawing::{draw_filled_circle_mut, draw_filled_rect_mut, draw_line_segment_mut};
use imageproc::rect::Rect;
use std::fs;
use std::path::Path;

fn main() {
    println!("Generating test images...");
    
    // Create test_data/images directory if it doesn't exist
    let images_dir = Path::new("test_data/images");
    fs::create_dir_all(images_dir).expect("Failed to create images directory");
    
    // Generate main test image
    generate_test_image();
    
    // Generate additional test images
    generate_simple_shapes();
    generate_pattern_image();
    generate_gradient_image();
    
    println!("Test images generated successfully!");
}

fn generate_test_image() {
    // Create a new 800x600 RGB image with white background
    let mut img = RgbImage::from_pixel(800, 600, Rgb([255u8, 255u8, 255u8]));
    
    // Draw a red rectangle
    draw_filled_rect_mut(
        &mut img,
        Rect::at(50, 50).of_size(150, 150),
        Rgb([255u8, 0u8, 0u8])
    );
    
    // Draw a blue circle
    draw_filled_circle_mut(
        &mut img,
        (325, 125),
        75,
        Rgb([0u8, 0u8, 255u8])
    );
    
    // Draw a green triangle using lines
    let green = Rgb([0u8, 255u8, 0u8]);
    // Draw triangle outline
    draw_line_segment_mut(&mut img, (500.0, 50.0), (600.0, 200.0), green);
    draw_line_segment_mut(&mut img, (600.0, 200.0), (450.0, 200.0), green);
    draw_line_segment_mut(&mut img, (450.0, 200.0), (500.0, 50.0), green);
    
    // Fill triangle manually (simple scanline fill)
    for y in 50..200 {
        let progress = (y - 50) as f32 / 150.0;
        let left_x = 500.0 - progress * 50.0;
        let right_x = 500.0 + progress * 100.0;
        for x in left_x as u32..right_x as u32 {
            if x < 800 && y < 600 {
                img.put_pixel(x, y as u32, green);
            }
        }
    }
    
    // Add text using simple lines (create "TEST" text)
    draw_text_simple(&mut img, 50, 300, "TEST", Rgb([0u8, 0u8, 0u8]), 3.0);
    
    // Add some additional elements for richer testing
    // Draw smaller colored squares
    for i in 0..5 {
        let x = 50 + i * 100;
        let color = match i {
            0 => Rgb([255u8, 128u8, 0u8]),   // Orange
            1 => Rgb([255u8, 0u8, 255u8]),   // Magenta
            2 => Rgb([0u8, 255u8, 255u8]),   // Cyan
            3 => Rgb([128u8, 0u8, 128u8]),   // Purple
            _ => Rgb([255u8, 255u8, 0u8]),   // Yellow
        };
        draw_filled_rect_mut(
            &mut img,
            Rect::at(x as i32, 500).of_size(50, 50),
            color
        );
    }
    
    // Save the image
    img.save("test_data/images/test_image.png").expect("Failed to save test image");
    println!("Generated: test_data/images/test_image.png");
}

fn generate_simple_shapes() {
    let mut img = RgbImage::from_pixel(400, 400, Rgb([240u8, 240u8, 240u8]));
    
    // Draw a grid of shapes
    draw_filled_rect_mut(&mut img, Rect::at(50, 50).of_size(100, 100), Rgb([255u8, 0u8, 0u8]));
    draw_filled_circle_mut(&mut img, (250, 100), 50, Rgb([0u8, 255u8, 0u8]));
    draw_filled_rect_mut(&mut img, Rect::at(50, 250).of_size(100, 100), Rgb([0u8, 0u8, 255u8]));
    draw_filled_circle_mut(&mut img, (250, 300), 50, Rgb([255u8, 255u8, 0u8]));
    
    // Add some lines
    for i in 0..10 {
        let y = 20 + i * 38;
        draw_line_segment_mut(&mut img, (10.0, y as f32), (390.0, y as f32), Rgb([200u8, 200u8, 200u8]));
    }
    
    img.save("test_data/images/simple_shapes.png").expect("Failed to save simple shapes");
    println!("Generated: test_data/images/simple_shapes.png");
}

fn generate_pattern_image() {
    let mut img = RgbImage::from_pixel(600, 200, Rgb([255u8, 255u8, 255u8]));
    
    // Create a checkerboard pattern
    for x in 0..30 {
        for y in 0..10 {
            if (x + y) % 2 == 0 {
                draw_filled_rect_mut(
                    &mut img,
                    Rect::at(x * 20, y * 20).of_size(20, 20),
                    Rgb([0u8, 0u8, 0u8])
                );
            }
        }
    }
    
    // Add some diagonal lines
    for i in 0..20 {
        let offset = i * 30;
        draw_line_segment_mut(&mut img, (offset as f32, 0.0), ((offset + 200) as f32, 200.0), Rgb([255u8, 0u8, 0u8]));
    }
    
    img.save("test_data/images/pattern.png").expect("Failed to save pattern");
    println!("Generated: test_data/images/pattern.png");
}

fn generate_gradient_image() {
    let width = 400;
    let height = 300;
    let mut img = ImageBuffer::new(width, height);
    
    // Create a gradient effect
    for x in 0..width {
        for y in 0..height {
            let r = (x as f32 / width as f32 * 255.0) as u8;
            let g = (y as f32 / height as f32 * 255.0) as u8;
            let b = ((x + y) as f32 / (width + height) as f32 * 255.0) as u8;
            img.put_pixel(x, y, Rgb([r, g, b]));
        }
    }
    
    // Add some circles on top
    for i in 0..5 {
        let x = 50 + i * 70;
        let y = 150;
        let radius = 20 + i * 5;
        draw_filled_circle_mut(&mut img, (x as i32, y), radius as i32, Rgb([255u8, 255u8, 255u8]));
    }
    
    img.save("test_data/images/gradient.png").expect("Failed to save gradient");
    println!("Generated: test_data/images/gradient.png");
}

// Simple text drawing using lines (creates basic letters)
fn draw_text_simple(img: &mut RgbImage, x: u32, y: u32, text: &str, color: Rgb<u8>, scale: f32) {
    let mut offset_x = x;
    
    for ch in text.chars() {
        match ch {
            'T' => {
                // Top horizontal line
                draw_line_segment_mut(img, (offset_x as f32, y as f32), ((offset_x + 20) as f32 * scale, y as f32), color);
                // Vertical line
                draw_line_segment_mut(img, ((offset_x + 10) as f32 * scale, y as f32), ((offset_x + 10) as f32 * scale, (y + 30) as f32 * scale), color);
            }
            'E' => {
                // Vertical line
                draw_line_segment_mut(img, (offset_x as f32, y as f32), (offset_x as f32, (y + 30) as f32 * scale), color);
                // Top horizontal
                draw_line_segment_mut(img, (offset_x as f32, y as f32), ((offset_x + 20) as f32 * scale, y as f32), color);
                // Middle horizontal
                draw_line_segment_mut(img, (offset_x as f32, (y + 15) as f32 * scale), ((offset_x + 15) as f32 * scale, (y + 15) as f32 * scale), color);
                // Bottom horizontal
                draw_line_segment_mut(img, (offset_x as f32, (y + 30) as f32 * scale), ((offset_x + 20) as f32 * scale, (y + 30) as f32 * scale), color);
            }
            'S' => {
                // Top curve
                draw_line_segment_mut(img, ((offset_x + 5) as f32 * scale, y as f32), ((offset_x + 20) as f32 * scale, y as f32), color);
                draw_line_segment_mut(img, (offset_x as f32, (y + 5) as f32 * scale), (offset_x as f32, (y + 10) as f32 * scale), color);
                // Middle
                draw_line_segment_mut(img, (offset_x as f32, (y + 15) as f32 * scale), ((offset_x + 20) as f32 * scale, (y + 15) as f32 * scale), color);
                // Bottom curve
                draw_line_segment_mut(img, ((offset_x + 20) as f32 * scale, (y + 20) as f32 * scale), ((offset_x + 20) as f32 * scale, (y + 25) as f32 * scale), color);
                draw_line_segment_mut(img, (offset_x as f32, (y + 30) as f32 * scale), ((offset_x + 15) as f32 * scale, (y + 30) as f32 * scale), color);
            }
            _ => {}
        }
        offset_x += (25.0 * scale) as u32;
    }
}