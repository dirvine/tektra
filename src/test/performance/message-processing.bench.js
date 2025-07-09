import { bench, describe } from 'vitest'

// Mock DOM for benchmarking
global.document = {
  createElement: () => ({
    textContent: '',
    get innerHTML() {
      return this.textContent
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;')
    }
  })
}

class BenchmarkProjectTektra {
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
}

describe('Message Processing Performance', () => {
  const app = new BenchmarkProjectTektra()
  
  // Test data
  const shortText = 'Hello world!'
  const mediumText = `This is a **bold** statement with *italic* text and some \`inline code\`.
  
  Here's a list:
  1. First item
  2. Second item
  3. Third item
  
  And a code block:
  \`\`\`javascript
  console.log('Hello, world!');
  \`\`\`
  `
  
  const longText = `# Large Document
  
  This is a very long document with multiple sections and various formatting.
  
  ## Section 1: Introduction
  
  Lorem ipsum dolor sit amet, consectetur adipiscing elit. **Sed do eiusmod** tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.
  
  ### Subsection 1.1
  
  Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. *Excepteur sint occaecat* cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.
  
  ## Section 2: Technical Details
  
  Here's some code:
  
  \`\`\`python
  def process_message(text):
      # Process the text
      formatted = escape_html(text)
      formatted = apply_markdown(formatted)
      return formatted
  \`\`\`
  
  And some inline code: \`formatResponse(text)\`
  
  ### Lists
  
  1. First numbered item
  2. Second numbered item
  3. Third numbered item with **bold** text
  
  - First bullet item
  - Second bullet item
  - Third bullet item with *italic* text
  
  ## Section 3: More Content
  
  Sed ut perspiciatis unde omnis iste natus error sit voluptatem accusantium doloremque laudantium, totam rem aperiam, eaque ipsa quae ab illo inventore veritatis et quasi architecto beatae vitae dicta sunt explicabo.
  
  \`\`\`javascript
  // Another code block
  class MessageProcessor {
    constructor() {
      this.formatters = [];
    }
    
    addFormatter(formatter) {
      this.formatters.push(formatter);
    }
    
    process(text) {
      return this.formatters.reduce((result, formatter) => {
        return formatter(result);
      }, text);
    }
  }
  \`\`\`
  
  Nemo enim ipsam voluptatem quia voluptas sit aspernatur aut odit aut fugit, sed quia consequuntur magni dolores eos qui ratione voluptatem sequi nesciunt.
  `
  
  const htmlInjectionText = `<script>alert('xss')</script>
  <img src=x onerror=alert('xss')>
  <svg onload=alert('xss')>
  <iframe src="javascript:alert('xss')">
  <object data="javascript:alert('xss')">
  <embed src="javascript:alert('xss')">
  <form><input type=submit onclick=alert('xss')>
  <table><tr><td onclick=alert('xss')>
  <div onmouseover=alert('xss')>
  <span onerror=alert('xss')>
  **Bold <script>alert('xss')</script> text**
  *Italic <img src=x onerror=alert('xss')> text*
  \`Code <svg onload=alert('xss')> block\`
  `
  
  const markdownHeavyText = `**Bold** *italic* \`code\` **more bold** *more italic* \`more code\`
  **Bold** *italic* \`code\` **more bold** *more italic* \`more code\`
  **Bold** *italic* \`code\` **more bold** *more italic* \`more code\`
  
  \`\`\`javascript
  // Code block 1
  const x = 1;
  \`\`\`
  
  \`\`\`python
  # Code block 2
  x = 1
  \`\`\`
  
  \`\`\`html
  <!-- Code block 3 -->
  <div>test</div>
  \`\`\`
  
  1. Item **bold** 1
  2. Item *italic* 2
  3. Item \`code\` 3
  4. Item **bold** 4
  5. Item *italic* 5
  
  - Bullet **bold** 1
  - Bullet *italic* 2
  - Bullet \`code\` 3
  - Bullet **bold** 4
  - Bullet *italic* 5
  `

  describe('HTML Escaping', () => {
    bench('escape short text', () => {
      app.escapeHtml(shortText)
    })
    
    bench('escape medium text', () => {
      app.escapeHtml(mediumText)
    })
    
    bench('escape long text', () => {
      app.escapeHtml(longText)
    })
    
    bench('escape HTML injection attempts', () => {
      app.escapeHtml(htmlInjectionText)
    })
  })

  describe('Message Formatting', () => {
    bench('format short message', () => {
      app.formatResponse(shortText)
    })
    
    bench('format medium message', () => {
      app.formatResponse(mediumText)
    })
    
    bench('format long message', () => {
      app.formatResponse(longText)
    })
    
    bench('format HTML injection attempts', () => {
      app.formatResponse(htmlInjectionText)
    })
    
    bench('format markdown-heavy text', () => {
      app.formatResponse(markdownHeavyText)
    })
  })

  describe('Repeated Operations', () => {
    bench('format 100 short messages', () => {
      for (let i = 0; i < 100; i++) {
        app.formatResponse(shortText)
      }
    })
    
    bench('format 10 medium messages', () => {
      for (let i = 0; i < 10; i++) {
        app.formatResponse(mediumText)
      }
    })
    
    bench('format 10 messages with XSS attempts', () => {
      for (let i = 0; i < 10; i++) {
        app.formatResponse(htmlInjectionText)
      }
    })
  })

  describe('Memory Usage', () => {
    bench('format message without memory leaks', () => {
      // Format and discard many messages to test for memory leaks
      for (let i = 0; i < 1000; i++) {
        const result = app.formatResponse(`Message ${i}: **bold** *italic* \`code\``)
        // Don't store the result to avoid artificial memory usage
      }
    })
    
    bench('escape text without memory leaks', () => {
      for (let i = 0; i < 1000; i++) {
        const result = app.escapeHtml(`<script>alert('${i}')</script>`)
        // Don't store the result
      }
    })
  })

  describe('Edge Cases', () => {
    bench('format empty string', () => {
      app.formatResponse('')
    })
    
    bench('format whitespace only', () => {
      app.formatResponse('   \n\n\t\t   ')
    })
    
    bench('format very long single line', () => {
      const longLine = 'a'.repeat(10000)
      app.formatResponse(longLine)
    })
    
    bench('format text with many line breaks', () => {
      const manyLines = 'line\n'.repeat(1000)
      app.formatResponse(manyLines)
    })
    
    bench('format nested markdown', () => {
      const nested = '**bold *italic* bold**'
      app.formatResponse(nested)
    })
    
    bench('format malformed markdown', () => {
      const malformed = '**unclosed bold *unclosed italic `unclosed code'
      app.formatResponse(malformed)
    })
  })

  describe('Real-world Scenarios', () => {
    bench('format typical chat message', () => {
      const chatMessage = 'Hi there! I need help with **JavaScript** arrays. Can you show me how to use `map()` and `filter()`?'
      app.formatResponse(chatMessage)
    })
    
    bench('format code explanation', () => {
      const codeExplanation = `Here's how to use \`map()\`:

      \`\`\`javascript
      const numbers = [1, 2, 3, 4, 5];
      const doubled = numbers.map(x => x * 2);
      console.log(doubled); // [2, 4, 6, 8, 10]
      \`\`\`

      The \`map()\` function **transforms** each element in an array.`
      app.formatResponse(codeExplanation)
    })
    
    bench('format AI assistant response', () => {
      const aiResponse = `I can help you with that! Here are the steps:

      1. **First step**: Initialize your project
      2. **Second step**: Install dependencies
      3. **Third step**: Configure your settings

      Let me know if you need more details about any of these steps.`
      app.formatResponse(aiResponse)
    })
    
    bench('format technical documentation', () => {
      const techDoc = `## API Reference

      ### \`sendMessage(text)\`

      Sends a message to the AI assistant.

      **Parameters:**
      - \`text\` (string): The message text

      **Returns:**
      - Promise<string>: The AI response

      **Example:**
      \`\`\`javascript
      const response = await sendMessage('Hello!');
      console.log(response);
      \`\`\`

      *Note: This function is asynchronous.*`
      app.formatResponse(techDoc)
    })
  })
})