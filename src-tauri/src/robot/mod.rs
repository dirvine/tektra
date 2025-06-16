use anyhow::Result;
use serde::{Serialize, Deserialize};
use tokio::net::TcpStream;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use std::collections::HashMap;

#[derive(Debug, Serialize, Deserialize)]
pub struct RobotCommand {
    pub action: String,
    pub parameters: HashMap<String, f64>,
    pub timestamp: i64,
    pub safety_check: bool,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct RobotState {
    pub joint_positions: Vec<f64>,
    pub end_effector_pose: Vec<f64>,
    pub gripper_state: f64,
    pub battery_level: f64,
}

pub struct RobotController {
    connection: Option<TcpStream>,
    current_state: RobotState,
    fast_decoder: FastDecoder,
}

pub struct FastDecoder {
    token_to_action: HashMap<u32, String>,
}

impl FastDecoder {
    pub fn new() -> Self {
        let mut token_to_action = HashMap::new();
        // Initialize FAST token mappings
        // These would be loaded from the FAST processor model
        token_to_action.insert(32, "move_forward".to_string());
        token_to_action.insert(44, "turn_left".to_string());
        token_to_action.insert(67, "grasp".to_string());
        token_to_action.insert(18, "release".to_string());
        
        Self { token_to_action }
    }
    
    pub fn decode_tokens(&self, tokens: &[u32]) -> Vec<String> {
        tokens.iter()
            .filter_map(|token| self.token_to_action.get(token).cloned())
            .collect()
    }
}

impl RobotController {
    pub fn new() -> Result<Self> {
        Ok(Self {
            connection: None,
            current_state: RobotState {
                joint_positions: vec![0.0; 7],
                end_effector_pose: vec![0.0; 6],
                gripper_state: 0.0,
                battery_level: 100.0,
            },
            fast_decoder: FastDecoder::new(),
        })
    }
    
    pub async fn connect(&mut self, address: &str) -> Result<()> {
        let stream = TcpStream::connect(address).await?;
        self.connection = Some(stream);
        Ok(())
    }
    
    pub async fn execute_action(&mut self, action_str: &str) -> Result<()> {
        // Parse FAST tokens from action string
        let tokens: Vec<u32> = action_str
            .split_whitespace()
            .filter_map(|s| s.parse().ok())
            .collect();
        
        // Decode tokens to actions
        let actions = self.fast_decoder.decode_tokens(&tokens);
        
        // Execute each action
        for action in actions {
            let command = self.create_command(&action)?;
            self.send_command(command).await?;
            
            // Wait for acknowledgment
            tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        }
        
        Ok(())
    }
    
    pub async fn send_command(&mut self, command: RobotCommand) -> Result<()> {
        if let Some(stream) = &mut self.connection {
            let json = serde_json::to_string(&command)?;
            stream.write_all(json.as_bytes()).await?;
            stream.write_all(b"\n").await?;
            Ok(())
        } else {
            // If not connected to real robot, log the command
            tracing::info!("Robot command (simulated): {:?}", command);
            Ok(())
        }
    }
    
    pub async fn update_state(&mut self) -> Result<()> {
        if let Some(stream) = &mut self.connection {
            // Request state update
            stream.write_all(b"GET_STATE\n").await?;
            
            // Read response
            let mut buffer = vec![0; 1024];
            let n = stream.read(&mut buffer).await?;
            
            if n > 0 {
                let response = String::from_utf8_lossy(&buffer[..n]);
                self.current_state = serde_json::from_str(&response)?;
            }
        }
        
        Ok(())
    }
    
    pub fn get_state(&self) -> &RobotState {
        &self.current_state
    }
    
    fn create_command(&self, action: &str) -> Result<RobotCommand> {
        let mut parameters = HashMap::new();
        
        match action {
            "move_forward" => {
                parameters.insert("distance".to_string(), 0.1);
                parameters.insert("speed".to_string(), 0.5);
            }
            "turn_left" => {
                parameters.insert("angle".to_string(), 15.0);
                parameters.insert("speed".to_string(), 0.3);
            }
            "grasp" => {
                parameters.insert("force".to_string(), 10.0);
                parameters.insert("width".to_string(), 0.05);
            }
            "release" => {
                parameters.insert("width".to_string(), 0.1);
            }
            _ => return Err(anyhow::anyhow!("Unknown action: {}", action)),
        }
        
        Ok(RobotCommand {
            action: action.to_string(),
            parameters,
            timestamp: chrono::Utc::now().timestamp(),
            safety_check: true,
        })
    }
}