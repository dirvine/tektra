// Simple 2D Avatar Implementation
export class Avatar2D {
    constructor(canvas) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.width = canvas.width;
        this.height = canvas.height;
        
        // Avatar state
        this.state = {
            expression: 'neutral',
            mouthOpen: 0,
            eyeBlink: 0,
            headTilt: 0,
            breathingPhase: 0
        };
        
        // Animation loop
        this.animationFrame = null;
        this.startAnimation();
        
        // Colors
        this.colors = {
            skin: '#FFD4A3',
            skinDark: '#F4B58A',
            eyes: '#4A90E2',
            mouth: '#E85D75',
            hair: '#3D3D3D'
        };
    }
    
    startAnimation() {
        const animate = () => {
            this.update();
            this.draw();
            this.animationFrame = requestAnimationFrame(animate);
        };
        animate();
    }
    
    stopAnimation() {
        if (this.animationFrame) {
            cancelAnimationFrame(this.animationFrame);
        }
    }
    
    update() {
        // Breathing animation
        this.state.breathingPhase += 0.05;
        
        // Random blinking
        if (Math.random() < 0.002 && this.state.eyeBlink === 0) {
            this.blink();
        }
    }
    
    draw() {
        const ctx = this.ctx;
        const cx = this.width / 2;
        const cy = this.height / 2;
        
        // Clear canvas
        ctx.clearRect(0, 0, this.width, this.height);
        
        // Apply breathing effect
        const breathScale = 1 + Math.sin(this.state.breathingPhase) * 0.02;
        
        ctx.save();
        ctx.translate(cx, cy);
        ctx.scale(breathScale, breathScale);
        ctx.rotate(this.state.headTilt * Math.PI / 180);
        ctx.translate(-cx, -cy);
        
        // Draw head
        this.drawHead(cx, cy);
        
        // Draw hair
        this.drawHair(cx, cy);
        
        // Draw eyes
        this.drawEyes(cx, cy);
        
        // Draw mouth
        this.drawMouth(cx, cy);
        
        // Draw expression-specific features
        this.drawExpression(cx, cy);
        
        ctx.restore();
    }
    
    drawHead(cx, cy) {
        const ctx = this.ctx;
        
        // Head shape
        ctx.fillStyle = this.colors.skin;
        ctx.beginPath();
        ctx.ellipse(cx, cy, 80, 100, 0, 0, Math.PI * 2);
        ctx.fill();
        
        // Neck
        ctx.fillRect(cx - 30, cy + 80, 60, 40);
        
        // Ears
        ctx.beginPath();
        ctx.ellipse(cx - 80, cy, 20, 30, -0.1, 0, Math.PI * 2);
        ctx.ellipse(cx + 80, cy, 20, 30, 0.1, 0, Math.PI * 2);
        ctx.fill();
    }
    
    drawHair(cx, cy) {
        const ctx = this.ctx;
        ctx.fillStyle = this.colors.hair;
        
        // Simple hair shape
        ctx.beginPath();
        ctx.arc(cx, cy - 50, 85, Math.PI, 0, false);
        ctx.bezierCurveTo(cx + 85, cy - 30, cx + 75, cy - 70, cx + 40, cy - 90);
        ctx.bezierCurveTo(cx + 20, cy - 95, cx - 20, cy - 95, cx - 40, cy - 90);
        ctx.bezierCurveTo(cx - 75, cy - 70, cx - 85, cy - 30, cx - 85, cy - 50);
        ctx.fill();
    }
    
    drawEyes(cx, cy) {
        const ctx = this.ctx;
        const eyeY = cy - 20;
        const eyeSpacing = 35;
        const blinkAmount = this.state.eyeBlink;
        
        // Left eye
        this.drawEye(cx - eyeSpacing, eyeY, blinkAmount);
        
        // Right eye
        this.drawEye(cx + eyeSpacing, eyeY, blinkAmount);
    }
    
    drawEye(x, y, blinkAmount) {
        const ctx = this.ctx;
        
        // Eye white
        ctx.fillStyle = 'white';
        ctx.beginPath();
        ctx.ellipse(x, y, 20, 15 * (1 - blinkAmount), 0, 0, Math.PI * 2);
        ctx.fill();
        
        if (blinkAmount < 0.8) {
            // Iris
            ctx.fillStyle = this.colors.eyes;
            ctx.beginPath();
            ctx.arc(x, y, 10, 0, Math.PI * 2);
            ctx.fill();
            
            // Pupil
            ctx.fillStyle = 'black';
            ctx.beginPath();
            ctx.arc(x, y, 5, 0, Math.PI * 2);
            ctx.fill();
            
            // Eye shine
            ctx.fillStyle = 'white';
            ctx.beginPath();
            ctx.arc(x + 3, y - 3, 2, 0, Math.PI * 2);
            ctx.fill();
        }
        
        // Eyelid
        ctx.strokeStyle = this.colors.skinDark;
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.ellipse(x, y, 20, 15 * (1 - blinkAmount), 0, 0, Math.PI * 2);
        ctx.stroke();
    }
    
    drawMouth(cx, cy) {
        const ctx = this.ctx;
        const mouthY = cy + 40;
        const mouthWidth = 40;
        const openAmount = this.state.mouthOpen;
        
        ctx.fillStyle = this.colors.mouth;
        ctx.strokeStyle = this.colors.skinDark;
        ctx.lineWidth = 2;
        
        if (openAmount > 0.1) {
            // Open mouth
            ctx.beginPath();
            ctx.ellipse(cx, mouthY, mouthWidth / 2, openAmount * 20, 0, 0, Math.PI * 2);
            ctx.fill();
            ctx.stroke();
            
            // Teeth (simple)
            if (openAmount > 0.3) {
                ctx.fillStyle = 'white';
                ctx.fillRect(cx - 15, mouthY - openAmount * 10, 30, 5);
            }
        } else {
            // Closed mouth (smile)
            ctx.beginPath();
            ctx.arc(cx, mouthY - 5, mouthWidth / 2, 0.2 * Math.PI, 0.8 * Math.PI);
            ctx.stroke();
        }
    }
    
    drawExpression(cx, cy) {
        const ctx = this.ctx;
        
        switch (this.state.expression) {
            case 'happy':
                // Blush
                ctx.fillStyle = 'rgba(255, 150, 150, 0.3)';
                ctx.beginPath();
                ctx.arc(cx - 60, cy + 10, 20, 0, Math.PI * 2);
                ctx.arc(cx + 60, cy + 10, 20, 0, Math.PI * 2);
                ctx.fill();
                break;
                
            case 'thinking':
                // Eyebrow raise
                ctx.strokeStyle = this.colors.hair;
                ctx.lineWidth = 3;
                ctx.beginPath();
                ctx.arc(cx - 35, cy - 45, 20, Math.PI * 1.2, Math.PI * 1.8);
                ctx.stroke();
                break;
                
            case 'surprised':
                // Wide eyes are handled in state
                break;
        }
    }
    
    // Public methods for animation control
    setState(newState) {
        Object.assign(this.state, newState);
    }
    
    setExpression(expression) {
        this.state.expression = expression;
        
        // Adjust features based on expression
        switch (expression) {
            case 'happy':
                this.state.mouthOpen = 0.3;
                break;
            case 'thinking':
                this.state.mouthOpen = 0;
                this.state.headTilt = 5;
                break;
            case 'surprised':
                this.state.mouthOpen = 0.5;
                break;
            default:
                this.state.mouthOpen = 0;
                this.state.headTilt = 0;
        }
    }
    
    blink() {
        const blinkDuration = 150;
        const startTime = Date.now();
        
        const animateBlink = () => {
            const elapsed = Date.now() - startTime;
            const progress = elapsed / blinkDuration;
            
            if (progress < 0.5) {
                this.state.eyeBlink = progress * 2;
            } else if (progress < 1) {
                this.state.eyeBlink = 2 - progress * 2;
            } else {
                this.state.eyeBlink = 0;
                return;
            }
            
            requestAnimationFrame(animateBlink);
        };
        
        animateBlink();
    }
    
    animateLipSync(frames) {
        let currentFrame = 0;
        const startTime = Date.now();
        
        const animate = () => {
            if (currentFrame >= frames.length) {
                this.state.mouthOpen = 0;
                return;
            }
            
            const elapsed = (Date.now() - startTime) / 1000;
            
            // Find the current frame based on timestamp
            while (currentFrame < frames.length - 1 && frames[currentFrame + 1].timestamp <= elapsed) {
                currentFrame++;
            }
            
            if (currentFrame < frames.length) {
                this.state.mouthOpen = frames[currentFrame].mouth_open;
                requestAnimationFrame(animate);
            }
        };
        
        animate();
    }
}