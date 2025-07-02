import React from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

const TestMarkdown: React.FC = () => {
  const testContent = `**What happens?** Normally, atoms have positively charged nuclei, and like charges repel. To fuse, the nuclei need to overcome this repulsion and get incredibly close. This requires extremely high temperatures and pressures.
* **Where does it happen?** The most common example of fusion is what powers the Sun and other stars. The intense gravity and heat in the core of these stars create the conditions necessary for fusion to occur.
* **What's the energy release?** When the nuclei fuse, the mass of the resulting nucleus is slightly *less* than the combined mass of the original nuclei. That "missing" mass is converted into energy according to Einstein's famous equation, E=mc². And that energy is *huge*!
* **Why is it important?** Fusion is considered a potentially clean and virtually limitless energy source. It doesn't produce greenhouse gases like fossil fuels, and the fuel (isotopes of hydrogen) is abundant.

**Challenges:**`;

  return (
    <div className="p-8 bg-gray-900 text-white min-h-screen">
      <h1 className="text-2xl font-bold mb-4">Markdown Test</h1>
      
      <div className="mb-8">
        <h2 className="text-xl mb-2">Raw Content:</h2>
        <pre className="bg-gray-800 p-4 rounded overflow-auto text-sm">{testContent}</pre>
      </div>
      
      <div>
        <h2 className="text-xl mb-2">Rendered with ReactMarkdown:</h2>
        <div className="bg-gray-800 p-4 rounded">
          <ReactMarkdown 
            remarkPlugins={[remarkGfm]}
            className="prose prose-invert max-w-none"
          >
            {testContent}
          </ReactMarkdown>
        </div>
      </div>
    </div>
  );
};

export default TestMarkdown;