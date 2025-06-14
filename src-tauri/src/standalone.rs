use rust_embed::RustEmbed;
use axum::{
    extract::Path,
    http::{header, StatusCode},
    response::{Html, IntoResponse, Response},
    routing::get,
    Router,
};
use std::sync::Arc;
use tower_http::cors::CorsLayer;
use tao::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};
use wry::WebViewBuilder;
use anyhow::Result;

#[derive(RustEmbed)]
#[folder = "dist/"]
struct Assets;

pub struct WebServer {
    port: u16,
}

impl WebServer {
    pub fn new() -> Self {
        Self { port: 3333 }
    }

    pub async fn start(&self) -> Result<()> {
        let app = Router::new()
            .route("/", get(serve_index))
            .route("/assets/*file", get(serve_static))
            .route("/*file", get(serve_static))
            .layer(CorsLayer::permissive());

        let addr = format!("127.0.0.1:{}", self.port);
        let listener = tokio::net::TcpListener::bind(&addr).await?;
        
        tracing::info!("Web server starting on http://{}", addr);
        
        axum::serve(listener, app).await?;
        Ok(())
    }

    pub fn get_url(&self) -> String {
        format!("http://127.0.0.1:{}", self.port)
    }
}

async fn serve_index() -> impl IntoResponse {
    serve_static(Path("index.html".to_string())).await
}

async fn serve_static(Path(path): Path<String>) -> impl IntoResponse {
    let path = path.trim_start_matches('/');
    
    match Assets::get(path) {
        Some(content) => {
            let mime = mime_guess::from_path(path).first_or_octet_stream();
            Response::builder()
                .status(StatusCode::OK)
                .header(header::CONTENT_TYPE, mime.as_ref())
                .body(content.data.into())
                .unwrap()
        }
        None => Response::builder()
            .status(StatusCode::NOT_FOUND)
            .body("404 Not Found".into())
            .unwrap(),
    }
}

pub fn create_window(url: &str) -> Result<()> {
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_title("Tektra AI Assistant")
        .with_inner_size(tao::dpi::LogicalSize::new(1200, 800))
        .build(&event_loop)?;

    let _webview = WebViewBuilder::new(&window)
        .with_url(url)
        .build()?;

    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Wait;

        match event {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => *control_flow = ControlFlow::Exit,
            _ => {}
        }
    });
}