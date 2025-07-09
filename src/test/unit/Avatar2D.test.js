import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'

// Mock canvas and context for testing
const mockContext = {
  clearRect: vi.fn(),
  fillRect: vi.fn(),
  strokeRect: vi.fn(),
  arc: vi.fn(),
  ellipse: vi.fn(),
  beginPath: vi.fn(),
  fill: vi.fn(),
  stroke: vi.fn(),
  closePath: vi.fn(),
  save: vi.fn(),
  restore: vi.fn(),
  translate: vi.fn(),
  scale: vi.fn(),
  rotate: vi.fn(),
  bezierCurveTo: vi.fn(),
  set fillStyle(value) { this._fillStyle = value },
  get fillStyle() { return this._fillStyle },
  set strokeStyle(value) { this._strokeStyle = value },
  get strokeStyle() { return this._strokeStyle },
  set lineWidth(value) { this._lineWidth = value },
  get lineWidth() { return this._lineWidth }
}

const mockCanvas = {
  width: 200,
  height: 200,
  getContext: vi.fn(() => mockContext)
}

// Mock requestAnimationFrame and cancelAnimationFrame
global.requestAnimationFrame = vi.fn((cb) => {
  const id = Math.random()
  setTimeout(cb, 16)
  return id
})

global.cancelAnimationFrame = vi.fn()

// Avatar2D class extracted from avatar.js for testing
class Avatar2D {
  constructor(canvas) {
    this.canvas = canvas
    this.ctx = canvas.getContext('2d')
    this.width = canvas.width
    this.height = canvas.height
    
    this.state = {
      expression: 'neutral',
      mouthOpen: 0,
      eyeBlink: 0,
      headTilt: 0,
      breathingPhase: 0
    }
    
    this.animationFrame = null
    this.startAnimation()
    
    this.colors = {
      skin: '#FFD4A3',
      skinDark: '#F4B58A',
      eyes: '#4A90E2',
      mouth: '#E85D75',
      hair: '#3D3D3D'
    }
  }
  
  startAnimation() {
    const animate = () => {
      this.update()
      this.draw()
      this.animationFrame = requestAnimationFrame(animate)
    }
    animate()
  }
  
  stopAnimation() {
    if (this.animationFrame) {
      cancelAnimationFrame(this.animationFrame)
      this.animationFrame = null
    }
  }
  
  update() {
    this.state.breathingPhase += 0.05
    
    if (Math.random() < 0.002 && this.state.eyeBlink === 0) {
      this.blink()
    }
  }
  
  draw() {
    const ctx = this.ctx
    const cx = this.width / 2
    const cy = this.height / 2
    
    ctx.clearRect(0, 0, this.width, this.height)
    
    const breathScale = 1 + Math.sin(this.state.breathingPhase) * 0.02
    
    ctx.save()
    ctx.translate(cx, cy)
    ctx.scale(breathScale, breathScale)
    ctx.rotate(this.state.headTilt * Math.PI / 180)
    ctx.translate(-cx, -cy)
    
    this.drawHead(cx, cy)
    this.drawHair(cx, cy)
    this.drawEyes(cx, cy)
    this.drawMouth(cx, cy)
    this.drawExpression(cx, cy)
    
    ctx.restore()
  }
  
  drawHead(cx, cy) {
    const ctx = this.ctx
    ctx.fillStyle = this.colors.skin
    ctx.beginPath()
    ctx.ellipse(cx, cy, 80, 100, 0, 0, Math.PI * 2)
    ctx.fill()
    ctx.fillRect(cx - 30, cy + 80, 60, 40)
    
    ctx.beginPath()
    ctx.ellipse(cx - 80, cy, 20, 30, -0.1, 0, Math.PI * 2)
    ctx.ellipse(cx + 80, cy, 20, 30, 0.1, 0, Math.PI * 2)
    ctx.fill()
  }
  
  drawHair(cx, cy) {
    const ctx = this.ctx
    ctx.fillStyle = this.colors.hair
    ctx.beginPath()
    ctx.arc(cx, cy - 50, 85, Math.PI, 0, false)
    ctx.bezierCurveTo(cx + 85, cy - 30, cx + 75, cy - 70, cx + 40, cy - 90)
    ctx.bezierCurveTo(cx + 20, cy - 95, cx - 20, cy - 95, cx - 40, cy - 90)
    ctx.bezierCurveTo(cx - 75, cy - 70, cx - 85, cy - 30, cx - 85, cy - 50)
    ctx.fill()
  }
  
  drawEyes(cx, cy) {
    const eyeY = cy - 20
    const eyeSpacing = 35
    const blinkAmount = this.state.eyeBlink
    
    this.drawEye(cx - eyeSpacing, eyeY, blinkAmount)
    this.drawEye(cx + eyeSpacing, eyeY, blinkAmount)
  }
  
  drawEye(x, y, blinkAmount) {
    const ctx = this.ctx
    
    ctx.fillStyle = 'white'
    ctx.beginPath()
    ctx.ellipse(x, y, 20, 15 * (1 - blinkAmount), 0, 0, Math.PI * 2)
    ctx.fill()
    
    if (blinkAmount < 0.8) {
      ctx.fillStyle = this.colors.eyes
      ctx.beginPath()
      ctx.arc(x, y, 10, 0, Math.PI * 2)
      ctx.fill()
      
      ctx.fillStyle = 'black'
      ctx.beginPath()
      ctx.arc(x, y, 5, 0, Math.PI * 2)
      ctx.fill()
      
      ctx.fillStyle = 'white'
      ctx.beginPath()
      ctx.arc(x + 3, y - 3, 2, 0, Math.PI * 2)
      ctx.fill()
    }
    
    ctx.strokeStyle = this.colors.skinDark
    ctx.lineWidth = 2
    ctx.beginPath()
    ctx.ellipse(x, y, 20, 15 * (1 - blinkAmount), 0, 0, Math.PI * 2)
    ctx.stroke()
  }
  
  drawMouth(cx, cy) {
    const ctx = this.ctx
    const mouthY = cy + 40
    const mouthWidth = 40
    const openAmount = this.state.mouthOpen
    
    ctx.fillStyle = this.colors.mouth
    ctx.strokeStyle = this.colors.skinDark
    ctx.lineWidth = 2
    
    if (openAmount > 0.1) {
      ctx.beginPath()
      ctx.ellipse(cx, mouthY, mouthWidth / 2, openAmount * 20, 0, 0, Math.PI * 2)
      ctx.fill()
      ctx.stroke()
      
      if (openAmount > 0.3) {
        ctx.fillStyle = 'white'
        ctx.fillRect(cx - 15, mouthY - openAmount * 10, 30, 5)
      }
    } else {
      ctx.beginPath()
      ctx.arc(cx, mouthY - 5, mouthWidth / 2, 0.2 * Math.PI, 0.8 * Math.PI)
      ctx.stroke()
    }
  }
  
  drawExpression(cx, cy) {
    const ctx = this.ctx
    
    switch (this.state.expression) {
      case 'happy':
        ctx.fillStyle = 'rgba(255, 150, 150, 0.3)'
        ctx.beginPath()
        ctx.arc(cx - 60, cy + 10, 20, 0, Math.PI * 2)
        ctx.arc(cx + 60, cy + 10, 20, 0, Math.PI * 2)
        ctx.fill()
        break
        
      case 'thinking':
        ctx.strokeStyle = this.colors.hair
        ctx.lineWidth = 3
        ctx.beginPath()
        ctx.arc(cx - 35, cy - 45, 20, Math.PI * 1.2, Math.PI * 1.8)
        ctx.stroke()
        break
    }
  }
  
  setState(newState) {
    Object.assign(this.state, newState)
  }
  
  setExpression(expression) {
    this.state.expression = expression
    
    switch (expression) {
      case 'happy':
        this.state.mouthOpen = 0.3
        break
      case 'thinking':
        this.state.mouthOpen = 0
        this.state.headTilt = 5
        break
      case 'surprised':
        this.state.mouthOpen = 0.5
        break
      default:
        this.state.mouthOpen = 0
        this.state.headTilt = 0
    }
  }
  
  blink() {
    const blinkDuration = 150
    const startTime = Date.now()
    
    const animateBlink = () => {
      const elapsed = Date.now() - startTime
      const progress = elapsed / blinkDuration
      
      if (progress < 0.5) {
        this.state.eyeBlink = progress * 2
      } else if (progress < 1) {
        this.state.eyeBlink = 2 - progress * 2
      } else {
        this.state.eyeBlink = 0
        return
      }
      
      requestAnimationFrame(animateBlink)
    }
    
    animateBlink()
  }
  
  animateLipSync(frames) {
    let currentFrame = 0
    const startTime = Date.now()
    
    const animate = () => {
      if (currentFrame >= frames.length) {
        this.state.mouthOpen = 0
        return
      }
      
      const elapsed = (Date.now() - startTime) / 1000
      
      while (currentFrame < frames.length - 1 && frames[currentFrame + 1].timestamp <= elapsed) {
        currentFrame++
      }
      
      if (currentFrame < frames.length) {
        this.state.mouthOpen = frames[currentFrame].mouth_open
        requestAnimationFrame(animate)
      }
    }
    
    animate()
  }
}

describe('Avatar2D', () => {
  let avatar

  beforeEach(() => {
    vi.clearAllMocks()
    global.requestAnimationFrame = vi.fn((cb) => {
      const id = Math.random()
      setTimeout(cb, 16)
      return id
    })
    
    avatar = new Avatar2D(mockCanvas)
  })

  afterEach(() => {
    if (avatar) {
      avatar.stopAnimation()
    }
    vi.clearAllMocks()
  })

  describe('Constructor', () => {
    it('should initialize with default state', () => {
      expect(avatar.width).toBe(200)
      expect(avatar.height).toBe(200)
      expect(avatar.state).toEqual({
        expression: 'neutral',
        mouthOpen: 0,
        eyeBlink: 0,
        headTilt: 0,
        breathingPhase: 0
      })
    })

    it('should initialize colors', () => {
      expect(avatar.colors).toEqual({
        skin: '#FFD4A3',
        skinDark: '#F4B58A',
        eyes: '#4A90E2',
        mouth: '#E85D75',
        hair: '#3D3D3D'
      })
    })

    it('should start animation loop', () => {
      expect(global.requestAnimationFrame).toHaveBeenCalled()
      expect(avatar.animationFrame).toBeDefined()
    })
  })

  describe('Animation Control', () => {
    it('should stop animation', () => {
      const frameId = avatar.animationFrame
      avatar.stopAnimation()
      
      expect(global.cancelAnimationFrame).toHaveBeenCalledWith(frameId)
      expect(avatar.animationFrame).toBeNull()
    })

    it('should update breathing phase', () => {
      const initialPhase = avatar.state.breathingPhase
      avatar.update()
      
      expect(avatar.state.breathingPhase).toBe(initialPhase + 0.05)
    })
  })

  describe('Drawing', () => {
    it('should clear canvas when drawing', () => {
      avatar.draw()
      
      expect(mockContext.clearRect).toHaveBeenCalledWith(0, 0, 200, 200)
    })

    it('should apply transformations', () => {
      avatar.draw()
      
      expect(mockContext.save).toHaveBeenCalled()
      expect(mockContext.translate).toHaveBeenCalled()
      expect(mockContext.scale).toHaveBeenCalled()
      expect(mockContext.rotate).toHaveBeenCalled()
      expect(mockContext.restore).toHaveBeenCalled()
    })

    it('should draw head', () => {
      avatar.drawHead(100, 100)
      
      expect(mockContext.fillStyle).toBe('#FFD4A3')
      expect(mockContext.ellipse).toHaveBeenCalledWith(100, 100, 80, 100, 0, 0, Math.PI * 2)
      expect(mockContext.fill).toHaveBeenCalled()
    })

    it('should draw hair', () => {
      avatar.drawHair(100, 100)
      
      expect(mockContext.fillStyle).toBe('#3D3D3D')
      expect(mockContext.arc).toHaveBeenCalled()
      expect(mockContext.bezierCurveTo).toHaveBeenCalled()
    })

    it('should draw eyes', () => {
      avatar.drawEyes(100, 100)
      
      expect(mockContext.ellipse).toHaveBeenCalled()
      expect(mockContext.arc).toHaveBeenCalled()
    })

    it('should draw mouth', () => {
      avatar.drawMouth(100, 100)
      
      expect(mockContext.fillStyle).toBe('#E85D75')
      expect(mockContext.strokeStyle).toBe('#F4B58A')
    })
  })

  describe('Expressions', () => {
    it('should set happy expression', () => {
      avatar.setExpression('happy')
      
      expect(avatar.state.expression).toBe('happy')
      expect(avatar.state.mouthOpen).toBe(0.3)
    })

    it('should set thinking expression', () => {
      avatar.setExpression('thinking')
      
      expect(avatar.state.expression).toBe('thinking')
      expect(avatar.state.mouthOpen).toBe(0)
      expect(avatar.state.headTilt).toBe(5)
    })

    it('should draw happy expression', () => {
      avatar.state.expression = 'happy'
      avatar.drawExpression(100, 100)
      
      expect(mockContext.fillStyle).toBe('rgba(255, 150, 150, 0.3)')
      expect(mockContext.arc).toHaveBeenCalledTimes(2)
    })
  })

  describe('State Management', () => {
    it('should update state', () => {
      const newState = {
        expression: 'happy',
        mouthOpen: 0.3,
        eyeBlink: 0.1
      }
      
      avatar.setState(newState)
      
      expect(avatar.state.expression).toBe('happy')
      expect(avatar.state.mouthOpen).toBe(0.3)
      expect(avatar.state.eyeBlink).toBe(0.1)
    })
  })

  describe('Animations', () => {
    it('should animate blink', () => {
      vi.spyOn(Date, 'now').mockReturnValue(1000)
      
      avatar.blink()
      
      expect(avatar.state.eyeBlink).toBeGreaterThan(0)
    })

    it('should animate lip sync', () => {
      const frames = [
        { timestamp: 0, mouth_open: 0.2 },
        { timestamp: 0.1, mouth_open: 0.5 }
      ]
      
      vi.spyOn(Date, 'now').mockReturnValue(1000)
      
      avatar.animateLipSync(frames)
      
      expect(avatar.state.mouthOpen).toBeGreaterThan(0)
    })
  })
})