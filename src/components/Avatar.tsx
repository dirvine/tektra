import React, { useRef, useEffect, useState } from 'react';
import { Bot, Volume2, VolumeX } from 'lucide-react';

interface AvatarProps {
  isListening?: boolean;
  isSpeaking?: boolean;
  expression?: 'neutral' | 'happy' | 'thinking' | 'surprised';
  size?: number;
}

const Avatar: React.FC<AvatarProps> = ({
  isListening = false,
  isSpeaking = false,
  expression = 'neutral',
  size = 120
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationRef = useRef<number>();
  const [time, setTime] = useState(0);

  useEffect(() => {
    const animate = () => {
      setTime(prev => prev + 0.1);
      animationRef.current = requestAnimationFrame(animate);
    };
    
    animationRef.current = requestAnimationFrame(animate);
    
    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const centerX = canvas.width / 2;
    const centerY = canvas.height / 2;
    const radius = size / 2 - 10;

    // Draw outer glow when listening
    if (isListening) {
      const glowRadius = radius + 15 + Math.sin(time * 3) * 5;
      const gradient = ctx.createRadialGradient(centerX, centerY, radius, centerX, centerY, glowRadius);
      gradient.addColorStop(0, 'rgba(59, 130, 246, 0.3)');
      gradient.addColorStop(1, 'rgba(59, 130, 246, 0)');
      
      ctx.fillStyle = gradient;
      ctx.fillRect(0, 0, canvas.width, canvas.height);
    }

    // Draw main circle (head)
    const headColor = isListening ? '#3b82f6' : '#1e293b';
    ctx.fillStyle = headColor;
    ctx.beginPath();
    ctx.arc(centerX, centerY, radius, 0, Math.PI * 2);
    ctx.fill();

    // Draw gradient overlay
    const headGradient = ctx.createRadialGradient(
      centerX - radius * 0.3, 
      centerY - radius * 0.3, 
      0,
      centerX, 
      centerY, 
      radius
    );
    headGradient.addColorStop(0, 'rgba(255, 255, 255, 0.2)');
    headGradient.addColorStop(1, 'rgba(0, 0, 0, 0.1)');
    
    ctx.fillStyle = headGradient;
    ctx.beginPath();
    ctx.arc(centerX, centerY, radius, 0, Math.PI * 2);
    ctx.fill();

    // Draw eyes
    const eyeY = centerY - radius * 0.2;
    const eyeRadius = radius * 0.08;
    const eyeSpacing = radius * 0.3;

    // Left eye
    ctx.fillStyle = '#ffffff';
    ctx.beginPath();
    ctx.arc(centerX - eyeSpacing, eyeY, eyeRadius, 0, Math.PI * 2);
    ctx.fill();

    // Right eye
    ctx.beginPath();
    ctx.arc(centerX + eyeSpacing, eyeY, eyeRadius, 0, Math.PI * 2);
    ctx.fill();

    // Eye pupils
    const pupilRadius = eyeRadius * 0.6;
    const blinkFactor = expression === 'thinking' ? Math.abs(Math.sin(time * 2)) : 1;
    
    ctx.fillStyle = '#1e293b';
    ctx.beginPath();
    ctx.arc(centerX - eyeSpacing, eyeY, pupilRadius * blinkFactor, 0, Math.PI * 2);
    ctx.fill();

    ctx.beginPath();
    ctx.arc(centerX + eyeSpacing, eyeY, pupilRadius * blinkFactor, 0, Math.PI * 2);
    ctx.fill();

    // Draw mouth based on expression and speaking state
    const mouthY = centerY + radius * 0.3;
    const mouthWidth = radius * 0.4;
    const mouthHeight = radius * 0.1;

    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 3;
    ctx.lineCap = 'round';

    if (isSpeaking) {
      // Animated mouth for speaking
      const mouthOpenness = (Math.sin(time * 8) + 1) * 0.5;
      ctx.fillStyle = '#000000';
      ctx.beginPath();
      ctx.ellipse(centerX, mouthY, mouthWidth * 0.6, mouthHeight * (0.5 + mouthOpenness), 0, 0, Math.PI * 2);
      ctx.fill();
    } else {
      // Static mouth based on expression
      ctx.beginPath();
      switch (expression) {
        case 'happy':
          ctx.arc(centerX, mouthY - mouthHeight, mouthWidth, 0.2, Math.PI - 0.2);
          break;
        case 'surprised':
          ctx.ellipse(centerX, mouthY, mouthWidth * 0.3, mouthHeight * 2, 0, 0, Math.PI * 2);
          break;
        case 'thinking':
          ctx.moveTo(centerX - mouthWidth * 0.3, mouthY);
          ctx.lineTo(centerX + mouthWidth * 0.3, mouthY);
          break;
        case 'neutral':
        default:
          ctx.arc(centerX, mouthY + mouthHeight, mouthWidth * 0.8, Math.PI + 0.3, Math.PI * 2 - 0.3);
          break;
      }
      ctx.stroke();
    }

    // Draw speech waves when speaking
    if (isSpeaking) {
      ctx.strokeStyle = 'rgba(59, 130, 246, 0.6)';
      ctx.lineWidth = 2;
      
      for (let i = 1; i <= 3; i++) {
        const waveRadius = radius + 20 + (i * 15);
        const opacity = 0.8 - (i * 0.2);
        const waveTime = time * 4 - (i * 0.5);
        
        ctx.globalAlpha = opacity * (0.5 + Math.sin(waveTime) * 0.5);
        ctx.beginPath();
        ctx.arc(centerX, centerY, waveRadius, 0, Math.PI * 2);
        ctx.stroke();
      }
      ctx.globalAlpha = 1;
    }

  }, [time, isListening, isSpeaking, expression, size]);

  return (
    <div className="avatar-container">
      <div className="avatar-wrapper">
        <canvas
          ref={canvasRef}
          width={size + 40}
          height={size + 40}
          className="avatar-canvas"
        />
        
        {/* Status indicators */}
        <div className="avatar-indicators">
          {isListening && (
            <div className="indicator listening">
              <Bot className="w-4 h-4" />
              <span>Listening</span>
            </div>
          )}
          
          {isSpeaking && (
            <div className="indicator speaking">
              <Volume2 className="w-4 h-4" />
              <span>Speaking</span>
            </div>
          )}
        </div>
      </div>
      
      <div className="avatar-info">
        <div className="avatar-name">Tektra</div>
        <div className="avatar-status">
          {isSpeaking ? 'Speaking...' : isListening ? 'Listening...' : 'Ready'}
        </div>
      </div>
    </div>
  );
};

export default Avatar;