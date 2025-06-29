export const formatResponse = (text: string): string => {
  // Escape HTML to prevent XSS
  const escapeHtml = (unsafe: string): string => {
    const div = document.createElement('div');
    div.textContent = unsafe;
    return div.innerHTML;
  };

  // Keep basic HTML escaping but preserve line breaks and formatting
  let formatted = escapeHtml(text);
  
  // Convert line breaks to <br> tags
  formatted = formatted.replace(/\n/g, '<br>');
  
  // Convert double line breaks to paragraph breaks
  formatted = formatted.replace(/(<br>){2,}/g, '</p><p>');
  
  // Wrap in paragraphs if there are paragraph breaks
  if (formatted.includes('</p><p>')) {
    formatted = '<p>' + formatted + '</p>';
  }
  
  // Handle basic markdown-style formatting
  // Bold text **text**
  formatted = formatted.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
  
  // Italic text *text*
  formatted = formatted.replace(/\*(.*?)\*/g, '<em>$1</em>');
  
  // Code blocks ```code```
  formatted = formatted.replace(/```(.*?)```/gs, '<pre><code>$1</code></pre>');
  
  // Inline code `code`
  formatted = formatted.replace(/`(.*?)`/g, '<code>$1</code>');
  
  // Handle numbered lists (1. item)
  formatted = formatted.replace(/^(\d+)\.\s(.+)$/gm, '<li>$2</li>');
  if (formatted.includes('<li>')) {
    formatted = formatted.replace(/(<li>.*<\/li>)/s, '<ol>$1</ol>');
  }
  
  // Handle bullet lists (- item or * item)
  formatted = formatted.replace(/^[-*]\s(.+)$/gm, '<li>$1</li>');
  if (formatted.includes('<li>') && !formatted.includes('<ol>')) {
    formatted = formatted.replace(/(<li>.*<\/li>)/s, '<ul>$1</ul>');
  }
  
  return formatted;
};