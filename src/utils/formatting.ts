export const formatResponse = (text: string): string => {
  // Escape HTML to prevent XSS
  function escapeHtml(unsafe: string): string {
    return unsafe
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }
  
  // Store code blocks and inline code temporarily
  const codeBlocks: string[] = [];
  const inlineCode: string[] = [];
  
  // First, extract code blocks to protect them
  let formatted = text.replace(/```([\s\S]*?)```/g, (match, code) => {
    const placeholder = `__CODE_BLOCK_${codeBlocks.length}__`;
    codeBlocks.push(`<pre><code>${escapeHtml(code.trim())}</code></pre>`);
    return placeholder;
  });
  
  // Extract inline code
  formatted = formatted.replace(/`([^`]+)`/g, (match, code) => {
    const placeholder = `__INLINE_CODE_${inlineCode.length}__`;
    inlineCode.push(`<code>${escapeHtml(code)}</code>`);
    return placeholder;
  });
  
  // Apply markdown transformations BEFORE escaping HTML
  // This way our HTML tags won't be escaped
  
  // Handle bold text **text**
  formatted = formatted.replace(/\*\*([^*]+)\*\*/g, (match, content) => {
    return `<strong>${escapeHtml(content)}</strong>`;
  });
  
  // Handle italic text *text* or _text_
  formatted = formatted.replace(/(?<!\*)\*([^*]+)\*(?!\*)/g, (match, content) => {
    return `<em>${escapeHtml(content)}</em>`;
  });
  formatted = formatted.replace(/_([^_]+)_/g, (match, content) => {
    return `<em>${escapeHtml(content)}</em>`;
  });
  
  // Handle headers
  formatted = formatted.replace(/^### (.+)$/gm, (match, heading) => {
    return `<h3>${escapeHtml(heading)}</h3>`;
  });
  formatted = formatted.replace(/^## (.+)$/gm, (match, heading) => {
    return `<h2>${escapeHtml(heading)}</h2>`;
  });
  formatted = formatted.replace(/^# (.+)$/gm, (match, heading) => {
    return `<h1>${escapeHtml(heading)}</h1>`;
  });
  
  // Handle lists
  const listItems: string[] = [];
  const numberedItems: string[] = [];
  
  // Bullet lists
  formatted = formatted.replace(/^[-*+] (.+)$/gm, (match, item) => {
    listItems.push(escapeHtml(item));
    return `__LIST_ITEM_${listItems.length - 1}__`;
  });
  
  // Numbered lists
  formatted = formatted.replace(/^\d+\. (.+)$/gm, (match, item) => {
    numberedItems.push(escapeHtml(item));
    return `__NUMBERED_ITEM_${numberedItems.length - 1}__`;
  });
  
  // Now escape any remaining HTML in the text
  // Split by our placeholders to avoid escaping them
  const parts = formatted.split(/(__(?:CODE_BLOCK|INLINE_CODE|LIST_ITEM|NUMBERED_ITEM)_\d+__)/);
  formatted = parts.map(part => {
    if (part.match(/^__(?:CODE_BLOCK|INLINE_CODE|LIST_ITEM|NUMBERED_ITEM)_\d+__$/)) {
      return part; // Keep placeholders as-is
    }
    return escapeHtml(part);
  }).join('');
  
  // Convert line breaks to <br> tags
  formatted = formatted.replace(/\n/g, '<br>');
  
  // Restore list items with proper HTML
  listItems.forEach((item, index) => {
    formatted = formatted.replace(`__LIST_ITEM_${index}__`, `<li>${item}</li>`);
  });
  numberedItems.forEach((item, index) => {
    formatted = formatted.replace(`__NUMBERED_ITEM_${index}__`, `<li>${item}</li>`);
  });
  
  // Wrap consecutive list items in ul/ol tags
  formatted = formatted.replace(/(<li>.*?<\/li>(<br>)?)+/g, (match) => {
    const items = match.replace(/<br>/g, '');
    // Check if it's a numbered list by looking for numbered item placeholders
    if (match.includes('__NUMBERED_ITEM_')) {
      return `<ol>${items}</ol>`;
    }
    return `<ul>${items}</ul>`;
  });
  
  // Restore code blocks and inline code
  codeBlocks.forEach((code, index) => {
    formatted = formatted.replace(`__CODE_BLOCK_${index}__`, code);
  });
  inlineCode.forEach((code, index) => {
    formatted = formatted.replace(`__INLINE_CODE_${index}__`, code);
  });
  
  // Handle paragraphs
  formatted = formatted.replace(/(<br>){2,}/g, '</p><p>');
  if (!formatted.startsWith('<h') && !formatted.startsWith('<ul') && !formatted.startsWith('<ol')) {
    formatted = '<p>' + formatted;
  }
  if (!formatted.endsWith('</h1>') && !formatted.endsWith('</h2>') && !formatted.endsWith('</h3>') && 
      !formatted.endsWith('</ul>') && !formatted.endsWith('</ol>') && !formatted.endsWith('</pre>')) {
    formatted = formatted + '</p>';
  }
  
  // Clean up any <br> tags right after block elements
  formatted = formatted.replace(/(<\/(?:h[1-6]|p|ul|ol|pre)>)<br>/g, '$1');
  
  return formatted;
};