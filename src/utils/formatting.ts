export const formatResponse = (text: string): string => {
  // Escape HTML to prevent XSS (but preserve our placeholders)
  function escapeHtml(unsafe: string): string {
    return unsafe
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }
  
  // First, handle code blocks to prevent them from being processed
  const codeBlocks: string[] = [];
  let formatted = text.replace(/```([\s\S]*?)```/g, (match, code) => {
    const placeholder = `__CODE_BLOCK_${codeBlocks.length}__`;
    codeBlocks.push(`<pre><code>${escapeHtml(code.trim())}</code></pre>`);
    return placeholder;
  });
  
  // Handle inline code
  const inlineCode: string[] = [];
  formatted = formatted.replace(/`([^`]+)`/g, (match, code) => {
    const placeholder = `__INLINE_CODE_${inlineCode.length}__`;
    inlineCode.push(`<code>${escapeHtml(code)}</code>`);
    return placeholder;
  });
  
  // Now escape the rest of the text
  formatted = escapeHtml(formatted);
  
  // Handle bold text **text** (must come before italic to avoid conflicts)
  formatted = formatted.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
  
  // Handle italic text *text* or _text_ (but not ** patterns)
  formatted = formatted.replace(/(?<!\*)\*([^*]+)\*(?!\*)/g, '<em>$1</em>');
  formatted = formatted.replace(/_([^_]+)_/g, '<em>$1</em>');
  
  // Handle headers
  formatted = formatted.replace(/^### (.+)$/gm, '<h3>$1</h3>');
  formatted = formatted.replace(/^## (.+)$/gm, '<h2>$1</h2>');
  formatted = formatted.replace(/^# (.+)$/gm, '<h1>$1</h1>');
  
  // Handle bullet lists (- item or * item at start of line)
  const listItems: string[] = [];
  formatted = formatted.replace(/^[-*+] (.+)$/gm, (match, item) => {
    listItems.push(item);
    return `__LIST_ITEM_${listItems.length - 1}__`;
  });
  
  // Handle numbered lists (1. item)
  const numberedItems: string[] = [];
  formatted = formatted.replace(/^\d+\. (.+)$/gm, (match, item) => {
    numberedItems.push(item);
    return `__NUMBERED_ITEM_${numberedItems.length - 1}__`;
  });
  
  // Convert line breaks to <br> tags (but not within list items)
  formatted = formatted.replace(/\n/g, '<br>');
  
  // Restore list items
  listItems.forEach((item, index) => {
    formatted = formatted.replace(`__LIST_ITEM_${index}__`, `<li>${item}</li>`);
  });
  numberedItems.forEach((item, index) => {
    formatted = formatted.replace(`__NUMBERED_ITEM_${index}__`, `<li>${item}</li>`);
  });
  
  // Wrap consecutive list items
  formatted = formatted.replace(/(<li>.*?<\/li>(<br>)?)+/g, (match) => {
    const items = match.replace(/<br>/g, '');
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
  
  // Handle paragraphs (double line breaks)
  formatted = formatted.replace(/(<br>){2,}/g, '</p><p>');
  if (formatted.includes('</p><p>') && !formatted.startsWith('<p>')) {
    formatted = '<p>' + formatted + '</p>';
  }
  
  // Clean up any <br> tags right after block elements
  formatted = formatted.replace(/(<\/(?:h[1-6]|p|ul|ol|pre)>)<br>/g, '$1');
  
  return formatted;
};