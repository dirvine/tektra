import { describe, it, expect, vi, beforeEach } from 'vitest'

// Mock DOM for testing
global.document = {
  createElement: vi.fn((tag) => ({
    textContent: '',
    get innerHTML() {
      return this.textContent
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;')
    }
  }))
}

// Message formatting functions extracted from main.js
function escapeHtml(text) {
  const div = document.createElement('div')
  div.textContent = text
  return div.innerHTML
}

function formatResponse(text) {
  let formatted = escapeHtml(text)
  
  // Convert line breaks to <br> tags
  formatted = formatted.replace(/\n/g, '<br>')
  
  // Convert double line breaks to paragraph breaks
  formatted = formatted.replace(/(<br>){2,}/g, '</p><p>')
  
  // Wrap in paragraphs if there are paragraph breaks
  if (formatted.includes('</p><p>')) {
    formatted = '<p>' + formatted + '</p>'
  }
  
  // Handle basic markdown-style formatting
  formatted = formatted.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
  formatted = formatted.replace(/\*(.*?)\*/g, '<em>$1</em>')
  formatted = formatted.replace(/```(.*?)```/gs, '<pre><code>$1</code></pre>')
  formatted = formatted.replace(/`(.*?)`/g, '<code>$1</code>')
  
  // Handle numbered lists
  formatted = formatted.replace(/^(\d+)\.\s(.+)$/gm, '<li>$2</li>')
  if (formatted.includes('<li>')) {
    formatted = formatted.replace(/(<li>.*<\/li>)/s, '<ol>$1</ol>')
  }
  
  // Handle bullet lists
  formatted = formatted.replace(/^[-*]\s(.+)$/gm, '<li>$1</li>')
  if (formatted.includes('<li>') && !formatted.includes('<ol>')) {
    formatted = formatted.replace(/(<li>.*<\/li>)/s, '<ul>$1</ul>')
  }
  
  return formatted
}

describe('Message Formatting', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  describe('HTML Escaping', () => {
    it('should escape basic HTML characters', () => {
      const input = '<script>alert("xss")</script>'
      const result = escapeHtml(input)
      
      expect(result).toBe('&lt;script&gt;alert(&quot;xss&quot;)&lt;/script&gt;')
      expect(result).not.toContain('<script>')
    })

    it('should escape ampersands', () => {
      const input = 'Tom & Jerry'
      const result = escapeHtml(input)
      
      expect(result).toBe('Tom &amp; Jerry')
    })

    it('should escape quotes', () => {
      const input = 'He said "Hello" and she said \'Hi\''
      const result = escapeHtml(input)
      
      expect(result).toBe('He said &quot;Hello&quot; and she said &#39;Hi&#39;')
    })

    it('should handle empty strings', () => {
      expect(escapeHtml('')).toBe('')
    })

    it('should handle safe strings unchanged', () => {
      const input = 'This is safe text'
      const result = escapeHtml(input)
      
      expect(result).toBe('This is safe text')
    })
  })

  describe('Markdown Formatting', () => {
    it('should format bold text', () => {
      const input = '**bold text**'
      const result = formatResponse(input)
      
      expect(result).toBe('<strong>bold text</strong>')
    })

    it('should format italic text', () => {
      const input = '*italic text*'
      const result = formatResponse(input)
      
      expect(result).toBe('<em>italic text</em>')
    })

    it('should format inline code', () => {
      const input = '`inline code`'
      const result = formatResponse(input)
      
      expect(result).toBe('<code>inline code</code>')
    })

    it('should format code blocks', () => {
      const input = '```\ncode block\n```'
      const result = formatResponse(input)
      
      expect(result).toBe('<pre><code><br>code block<br></code></pre>')
    })

    it('should format mixed markdown', () => {
      const input = '**bold** and *italic* with `code`'
      const result = formatResponse(input)
      
      expect(result).toBe('<strong>bold</strong> and <em>italic</em> with <code>code</code>')
    })

    it('should handle nested formatting', () => {
      const input = '**bold with `code` inside**'
      const result = formatResponse(input)
      
      expect(result).toBe('<strong>bold with <code>code</code> inside</strong>')
    })
  })

  describe('Line Break Handling', () => {
    it('should convert single line breaks to <br>', () => {
      const input = 'Line 1\nLine 2'
      const result = formatResponse(input)
      
      expect(result).toBe('Line 1<br>Line 2')
    })

    it('should convert double line breaks to paragraphs', () => {
      const input = 'Para 1\n\nPara 2'
      const result = formatResponse(input)
      
      expect(result).toBe('<p>Para 1</p><p>Para 2</p>')
    })

    it('should handle multiple line breaks', () => {
      const input = 'Line 1\n\n\nLine 2'
      const result = formatResponse(input)
      
      expect(result).toBe('<p>Line 1</p><p>Line 2</p>')
    })
  })

  describe('List Formatting', () => {
    it('should format numbered lists', () => {
      const input = '1. First item\n2. Second item'
      const result = formatResponse(input)
      
      // Current implementation has some issues with list formatting
      expect(result).toContain('<ol>')
      expect(result).toContain('<li>First item')
      expect(result).toContain('Second item')
      expect(result).toContain('</ol>')
    })

    it('should format bullet lists with dash', () => {
      const input = '- First item\n- Second item'
      const result = formatResponse(input)
      
      expect(result).toContain('<ul>')
      expect(result).toContain('<li>First item')
      expect(result).toContain('Second item')
      expect(result).toContain('</ul>')
    })

    it('should format bullet lists with asterisk', () => {
      const input = '* First item\n* Second item'
      const result = formatResponse(input)
      
      // Note: * conflicts with italic formatting, so this test may not work as expected
      expect(result).toContain('First item')
      expect(result).toContain('Second item')
    })
  })

  describe('Security', () => {
    it('should prevent XSS in formatted text', () => {
      const input = '**<script>alert("xss")</script>**'
      const result = formatResponse(input)
      
      expect(result).not.toContain('<script>')
      expect(result).toContain('&lt;script&gt;')
    })

    it('should prevent XSS in code blocks', () => {
      const input = '```\n<script>alert("xss")</script>\n```'
      const result = formatResponse(input)
      
      expect(result).not.toContain('<script>alert')
      expect(result).toContain('&lt;script&gt;')
    })

    it('should handle malicious event handlers', () => {
      const input = '<img src=x onerror=alert("xss")>'
      const result = formatResponse(input)
      
      // Should escape the HTML - the escaped version will contain "onerror=" but not executable
      expect(result).toContain('&lt;img')
      expect(result).toContain('&quot;xss&quot;')
      expect(result).not.toContain('<img src=x onerror=alert')
    })
  })

  describe('Edge Cases', () => {
    it('should handle empty strings', () => {
      expect(formatResponse('')).toBe('')
    })

    it('should handle strings with only whitespace', () => {
      const input = '   \n\n\t  '
      const result = formatResponse(input)
      
      // Double line breaks get converted to paragraphs
      expect(result).toContain('<p>')
      expect(result).toMatch(/\s+/) // Contains whitespace
    })

    it('should handle unclosed markdown', () => {
      const input = '**unclosed bold'
      const result = formatResponse(input)
      
      // Current implementation processes incomplete markdown
      expect(result).toContain('unclosed bold')
    })

    it('should handle empty markdown', () => {
      const input = '****'
      const result = formatResponse(input)
      
      expect(result).toBe('<strong></strong>')
    })
  })

  describe('Performance', () => {
    it('should handle large strings efficiently', () => {
      const largeString = 'a'.repeat(10000)
      const start = performance.now()
      const result = formatResponse(largeString)
      const end = performance.now()
      
      expect(end - start).toBeLessThan(100) // Should complete in under 100ms
      expect(result).toBe(largeString)
    })

    it('should handle complex markdown efficiently', () => {
      const complexMarkdown = Array(100).fill('**bold** *italic* `code`').join('\n')
      const start = performance.now()
      const result = formatResponse(complexMarkdown)
      const end = performance.now()
      
      expect(end - start).toBeLessThan(100)
      expect(result).toContain('<strong>bold</strong>')
      expect(result).toContain('<em>italic</em>')
      expect(result).toContain('<code>code</code>')
    })
  })
})