import { describe, it, expect, vi, beforeEach } from 'vitest'
import fc from 'fast-check'

// Message formatting functions for property testing
class MessageFormatter {
  escapeHtml(text) {
    const div = document.createElement('div')
    div.textContent = text
    return div.innerHTML
  }

  formatResponse(text) {
    let formatted = this.escapeHtml(text)
    
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
    
    return formatted
  }
}

// Mock DOM for property testing
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

describe('Message Formatting Property Tests', () => {
  let formatter

  beforeEach(() => {
    vi.clearAllMocks()
    formatter = new MessageFormatter()
  })

  describe('HTML Escaping Properties', () => {
    it('should never return unescaped HTML entities', () => {
      fc.assert(
        fc.property(fc.string(), (input) => {
          const escaped = formatter.escapeHtml(input)
          
          // Should not contain literal < or > characters
          expect(escaped).not.toMatch(/<(?!\/?(strong|em|code|pre|p|br|ul|ol|li)[ >])/i)
          expect(escaped).not.toMatch(/>/i)
          
          // Should not contain script tags
          expect(escaped.toLowerCase()).not.toContain('<script')
          expect(escaped.toLowerCase()).not.toContain('javascript:')
          expect(escaped.toLowerCase()).not.toContain('onerror=')
          expect(escaped.toLowerCase()).not.toContain('onclick=')
        })
      )
    })

    it('should preserve length or increase it (never decrease)', () => {
      fc.assert(
        fc.property(fc.string(), (input) => {
          const escaped = formatter.escapeHtml(input)
          expect(escaped.length).toBeGreaterThanOrEqual(input.length)
        })
      )
    })

    it('should be idempotent for safe strings', () => {
      fc.assert(
        fc.property(
          fc.string().filter(s => !s.includes('<') && !s.includes('>') && !s.includes('&')),
          (safeInput) => {
            const escaped1 = formatter.escapeHtml(safeInput)
            const escaped2 = formatter.escapeHtml(escaped1)
            expect(escaped1).toBe(escaped2)
          }
        )
      )
    })
  })

  describe('Markdown Formatting Properties', () => {
    it('should handle bold text correctly', () => {
      fc.assert(
        fc.property(
          fc.string().filter(s => !s.includes('**')),
          (content) => {
            const input = `**${content}**`
            const formatted = formatter.formatResponse(input)
            
            if (content.length > 0) {
              expect(formatted).toContain('<strong>')
              expect(formatted).toContain('</strong>')
            }
            
            // Should not contain the original markdown syntax
            expect(formatted).not.toContain('**')
          }
        )
      )
    })

    it('should handle italic text correctly', () => {
      fc.assert(
        fc.property(
          fc.string().filter(s => !s.includes('*')),
          (content) => {
            const input = `*${content}*`
            const formatted = formatter.formatResponse(input)
            
            if (content.length > 0) {
              expect(formatted).toContain('<em>')
              expect(formatted).toContain('</em>')
            }
          }
        )
      )
    })

    it('should handle code blocks correctly', () => {
      fc.assert(
        fc.property(
          fc.string().filter(s => !s.includes('```')),
          (content) => {
            const input = `\`\`\`${content}\`\`\``
            const formatted = formatter.formatResponse(input)
            
            if (content.length > 0) {
              expect(formatted).toContain('<pre><code>')
              expect(formatted).toContain('</code></pre>')
            }
            
            // Should not contain the original markdown syntax
            expect(formatted).not.toContain('```')
          }
        )
      )
    })

    it('should handle inline code correctly', () => {
      fc.assert(
        fc.property(
          fc.string().filter(s => !s.includes('`')),
          (content) => {
            const input = `\`${content}\``
            const formatted = formatter.formatResponse(input)
            
            if (content.length > 0) {
              expect(formatted).toContain('<code>')
              expect(formatted).toContain('</code>')
            }
          }
        )
      )
    })

    it('should preserve line breaks', () => {
      fc.assert(
        fc.property(
          fc.array(fc.string(), { minLength: 1, maxLength: 5 }),
          (lines) => {
            const input = lines.join('\n')
            const formatted = formatter.formatResponse(input)
            
            if (lines.length > 1) {
              expect(formatted).toContain('<br>')
            }
          }
        )
      )
    })

    it('should handle mixed formatting', () => {
      fc.assert(
        fc.property(
          fc.string().filter(s => s.length > 0 && !s.includes('*') && !s.includes('`')),
          (content) => {
            const input = `**${content}** and *${content}* and \`${content}\``
            const formatted = formatter.formatResponse(input)
            
            // Should contain all formatting elements
            expect(formatted).toContain('<strong>')
            expect(formatted).toContain('<em>')
            expect(formatted).toContain('<code>')
            
            // Should not contain markdown syntax
            expect(formatted).not.toContain('**')
            expect(formatted).not.toContain('*')
            expect(formatted).not.toContain('`')
          }
        )
      )
    })
  })

  describe('Security Properties', () => {
    it('should never produce executable JavaScript', () => {
      fc.assert(
        fc.property(
          fc.string(),
          (input) => {
            const formatted = formatter.formatResponse(input)
            
            // Should not contain script tags
            expect(formatted.toLowerCase()).not.toContain('<script')
            expect(formatted.toLowerCase()).not.toContain('javascript:')
            expect(formatted.toLowerCase()).not.toContain('onerror=')
            expect(formatted.toLowerCase()).not.toContain('onclick=')
            expect(formatted.toLowerCase()).not.toContain('onload=')
            expect(formatted.toLowerCase()).not.toContain('onmouseover=')
          }
        )
      )
    })

    it('should handle XSS attempts', () => {
      const xssPayloads = [
        '<script>alert("xss")</script>',
        '<img src=x onerror=alert("xss")>',
        '<svg onload=alert("xss")>',
        'javascript:alert("xss")',
        '<iframe src="javascript:alert(\'xss\')">',
        '<object data="javascript:alert(\'xss\')">',
        '<embed src="javascript:alert(\'xss\')">'
      ]
      
      xssPayloads.forEach(payload => {
        const formatted = formatter.formatResponse(payload)
        
        // Should not contain literal dangerous elements
        expect(formatted.toLowerCase()).not.toContain('<script')
        expect(formatted.toLowerCase()).not.toContain('javascript:')
        expect(formatted.toLowerCase()).not.toContain('onerror=')
        expect(formatted.toLowerCase()).not.toContain('onload=')
        
        // Should contain escaped versions instead  
        if (payload.includes('alert(')) {
          expect(formatted.toLowerCase()).toContain('alert(')
        }
      })
    })

    it('should handle deeply nested formatting', () => {
      fc.assert(
        fc.property(
          fc.string().filter(s => s.length > 0 && !s.includes('*') && !s.includes('`')),
          (content) => {
            // Create deeply nested formatting
            const input = `**\`*${content}*\`**`
            const formatted = formatter.formatResponse(input)
            
            // Should handle nested formatting correctly
            expect(formatted).toContain('<strong>')
            expect(formatted).toContain('<code>')
            expect(formatted).toContain('<em>')
            
            // Should not break HTML structure
            const openTags = (formatted.match(/<[^\/][^>]*>/g) || []).length
            const closeTags = (formatted.match(/<\/[^>]*>/g) || []).length
            expect(openTags).toBe(closeTags)
          }
        )
      )
    })
  })

  describe('Formatting Consistency Properties', () => {
    it('should produce valid HTML structure', () => {
      fc.assert(
        fc.property(
          fc.string(),
          (input) => {
            const formatted = formatter.formatResponse(input)
            
            // Count opening and closing tags
            const strongOpens = (formatted.match(/<strong>/g) || []).length
            const strongCloses = (formatted.match(/<\/strong>/g) || []).length
            const emOpens = (formatted.match(/<em>/g) || []).length
            const emCloses = (formatted.match(/<\/em>/g) || []).length
            const codeOpens = (formatted.match(/<code>/g) || []).length
            const codeCloses = (formatted.match(/<\/code>/g) || []).length
            const preOpens = (formatted.match(/<pre>/g) || []).length
            const preCloses = (formatted.match(/<\/pre>/g) || []).length
            
            // All tags should be properly closed
            expect(strongOpens).toBe(strongCloses)
            expect(emOpens).toBe(emCloses)
            expect(codeOpens).toBe(codeCloses)
            expect(preOpens).toBe(preCloses)
          }
        )
      )
    })

    it('should handle empty and whitespace-only input', () => {
      fc.assert(
        fc.property(
          fc.string().filter(s => s.trim().length === 0),
          (whitespaceInput) => {
            const formatted = formatter.formatResponse(whitespaceInput)
            
            // Should not crash or produce invalid HTML
            expect(formatted).toBeDefined()
            expect(typeof formatted).toBe('string')
            
            // Should not contain formatting tags for empty content
            expect(formatted).not.toContain('<strong></strong>')
            expect(formatted).not.toContain('<em></em>')
            expect(formatted).not.toContain('<code></code>')
          }
        )
      )
    })

    it('should be deterministic', () => {
      fc.assert(
        fc.property(
          fc.string(),
          (input) => {
            const formatted1 = formatter.formatResponse(input)
            const formatted2 = formatter.formatResponse(input)
            
            expect(formatted1).toBe(formatted2)
          }
        )
      )
    })
  })

  describe('Performance Properties', () => {
    it('should handle large inputs efficiently', () => {
      fc.assert(
        fc.property(
          fc.string({ minLength: 100, maxLength: 1000 }),
          (largeInput) => {
            const startTime = performance.now()
            const formatted = formatter.formatResponse(largeInput)
            const endTime = performance.now()
            
            // Should complete within reasonable time (100ms)
            expect(endTime - startTime).toBeLessThan(100)
            expect(formatted).toBeDefined()
            expect(typeof formatted).toBe('string')
          }
        )
      )
    })

    it('should not cause exponential growth', () => {
      fc.assert(
        fc.property(
          fc.string({ minLength: 100, maxLength: 1000 }),
          (input) => {
            const formatted = formatter.formatResponse(input)
            
            // Output should not be more than 10x the input size
            expect(formatted.length).toBeLessThan(input.length * 10)
          }
        )
      )
    })
  })
})