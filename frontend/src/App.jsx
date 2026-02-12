import { useState, useRef, useEffect } from 'react'
import './App.css'

const API_URL = 'http://localhost:8000'

function App() {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [sources, setSources] = useState([])
  const messagesEndRef = useRef(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const sendMessage = async (e) => {
    e.preventDefault()
    if (!input.trim() || isLoading) return

    const userMessage = { role: 'user', content: input.trim() }
    const newMessages = [...messages, userMessage]
    setMessages(newMessages)
    setInput('')
    setIsLoading(true)
    setSources([])

    try {
      const response = await fetch(`${API_URL}/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          messages: newMessages,
          top_k: 5,
        }),
      })

      if (!response.ok) {
        throw new Error('Failed to get response')
      }

      const data = await response.json()
      const assistantMessage = { role: 'assistant', content: data.answer }
      setMessages([...newMessages, assistantMessage])
      setSources(data.sources || [])
    } catch (error) {
      console.error('Error:', error)
      const errorMessage = {
        role: 'assistant',
        content: 'Sorry, I encountered an error. Please make sure the backend server is running.'
      }
      setMessages([...newMessages, errorMessage])
    } finally {
      setIsLoading(false)
    }
  }

  const clearChat = () => {
    setMessages([])
    setSources([])
  }

  return (
    <div className="chat-container">
      <header className="chat-header">
        <h1>RAG Chat Assistant</h1>
        <p className="subtitle">Ask questions about your documents</p>
        {messages.length > 0 && (
          <button className="clear-btn" onClick={clearChat}>
            Clear Chat
          </button>
        )}
      </header>

      <div className="messages-container">
        {messages.length === 0 ? (
          <div className="welcome-message">
            <h2>Welcome!</h2>
            <p>Start a conversation by typing a message below.</p>
            <p>I can help you explore and understand your uploaded documents.</p>
          </div>
        ) : (
          messages.map((msg, index) => (
            <div
              key={index}
              className={`message ${msg.role === 'user' ? 'user-message' : 'assistant-message'}`}
            >
              <div className="message-role">
                {msg.role === 'user' ? 'You' : 'Assistant'}
              </div>
              <div className="message-content">{msg.content}</div>
            </div>
          ))
        )}

        {isLoading && (
          <div className="message assistant-message">
            <div className="message-role">Assistant</div>
            <div className="message-content loading">
              <span className="dot"></span>
              <span className="dot"></span>
              <span className="dot"></span>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {sources.length > 0 && (
        <div className="sources-container">
          <strong>Sources:</strong>
          <ul>
            {sources.map((source, index) => (
              <li key={index}>{source}</li>
            ))}
          </ul>
        </div>
      )}

      <form className="input-container" onSubmit={sendMessage}>
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Type your message..."
          disabled={isLoading}
          autoFocus
        />
        <button type="submit" disabled={isLoading || !input.trim()}>
          {isLoading ? 'Sending...' : 'Send'}
        </button>
      </form>
    </div>
  )
}

export default App
