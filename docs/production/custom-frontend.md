# Custom Frontend

The fastest way to integrate Parlant into your React application is using our official [`parlant-chat-react`](https://github.com/emcie-co/parlant-chat-react) widget. This component provides a complete chat interface that connects directly to your Parlant agents.

### Installation and Basic Setup

Install the widget via npm or yarn:

```bash
npm install parlant-chat-react
# or
yarn add parlant-chat-react
```

Then integrate it into your React application:

```jsx
import React from 'react';
import ParlantChatbox from 'parlant-chat-react';

function App() {
  return (
    <div>
      <h1>My Application</h1>
      <ParlantChatbox
        server="http://localhost:8800"  // Your Parlant server URL
        agentId="your-agent-id"         // Your agent's ID
      />
    </div>
  );
}

export default App;
```

### Configuration Options

The widget supports several configuration props:

```jsx
<ParlantChatbox
  // Required props
  server="http://localhost:8800"
  agentId="your-agent-id"

  // Optional props
  sessionId="existing-session-id"     // Continue existing session
  customerId="customer-123"           // Associate with specific customer
  float={true}                        // Display as floating popup
  titleFn={(session) => `Chat ${session.id}`}  // Dynamic title generation
/>
```

### Common Customizations

#### Styling with Custom Classes

Customize the appearance using CSS class overrides:

```jsx
<ParlantChatbox
  server="http://localhost:8800"
  agentId="your-agent-id"
  classNames={{
    chatboxWrapper: "my-chat-wrapper",
    chatbox: "my-chatbox",
    messagesArea: "my-messages",
    agentMessage: "my-agent-bubble",
    customerMessage: "my-customer-bubble",
    textarea: "my-input-field",
    popupButton: "my-popup-btn"
  }}
/>
```

#### Custom Component Replacement

Replace specific components with your own:

```jsx
<ParlantChatbox
  server="http://localhost:8800"
  agentId="your-agent-id"
  components={{
    popupButton: ({ toggleChatOpen }) => (
      <button
        onClick={toggleChatOpen}
        className="custom-chat-button"
      >
        💬 Chat with us
      </button>
    ),
    agentMessage: ({ message }) => (
      <div className="custom-agent-message">
        <img src="https://parlant.io/agent-avatar.png" alt="Agent" />
        <p>{message.data.message}</p>
      </div>
    )
  }}
/>
```

#### Floating Chat Mode

Enable popup mode for a floating chat interface:

```jsx
<ParlantChatbox
  server="http://localhost:8800"
  agentId="your-agent-id"
  float={true}
  popupButton={<ChatIcon size={24} color="white" />}
/>
```

> **Reference Implementation**
>
> The parlant-chat-react widget is open source! You can [examine its implementation on GitHub](https://github.com/emcie-co/parlant-chat-react) as a reference for creating custom widgets in other UI frameworks like Vue, Angular, or vanilla JavaScript. The source code demonstrates best practices for session management, event handling, and UI state synchronization.

## Building a Custom Frontend

If you need more control than the React widget provides, or you're using a different framework, you can build a custom frontend using Parlant's client APIs directly.

### Step 1: Initialize the Parlant Client

Start by setting up the Parlant client to communicate with your server:

#### TypeScript

```typescript
import { ParlantClient } from 'parlant-client';

class ParlantChat {
  private client: ParlantClient;
  private sessionId: string | null = null;
  private lastOffset: number = 0;

  constructor(serverUrl: string) {
    this.client = new ParlantClient({
      environment: serverUrl
    });
  }
}
```

#### JavaScript

```javascript
import { ParlantClient } from 'parlant-client';

class ParlantChat {
  constructor(serverUrl) {
    this.client = new ParlantClient({
      environment: serverUrl
    });
    this.sessionId = null;
    this.lastOffset = 0;
  }
}
```

### Step 2: Create a Session

Initialize a conversation session with your agent:

#### TypeScript

```typescript
async createSession(agentId: string, customerId?: string): Promise<string> {
  try {
    const session = await this.client.sessions.create({
      agentId: agentId,
      customerId: customerId,
      title: `Chat Session ${new Date().toLocaleString()}`
    });

    this.sessionId = session.id;
    console.log('Session created:', this.sessionId);

    // Start monitoring for events
    this.startEventMonitoring();

    return this.sessionId;
  } catch (error) {
    console.error('Failed to create session:', error);
    throw error;
  }
}
```

#### JavaScript

```javascript
async createSession(agentId, customerId) {
  try {
    const session = await this.client.sessions.create({
      agentId: agentId,
      customerId: customerId,
      title: `Chat Session ${new Date().toLocaleString()}`
    });

    this.sessionId = session.id;
    console.log('Session created:', this.sessionId);

    // Start monitoring for events
    this.startEventMonitoring();

    return this.sessionId;
  } catch (error) {
    console.error('Failed to create session:', error);
    throw error;
  }
}
```

### Step 3: Send Customer Messages

Handle user input and send messages to the agent:

#### TypeScript

```typescript
async sendMessage(message: string): Promise<void> {
  if (!this.sessionId) {
    throw new Error('No active session');
  }

  try {
    await this.client.sessions.createEvent(this.sessionId, {
      kind: "message",
      source: "customer",
      message: message
    });

    // Message will appear in UI when it comes back from event monitoring
    console.log('Message sent:', message);
  } catch (error) {
    console.error('Failed to send message:', error);
    throw error;
  }
}
```

#### JavaScript

```javascript
async sendMessage(message) {
  if (!this.sessionId) {
    throw new Error('No active session');
  }

  try {
    await this.client.sessions.createEvent(this.sessionId, {
      kind: "message",
      source: "customer",
      message: message
    });

    // Message will appear in UI when it comes back from event monitoring
    console.log('Message sent:', message);
  } catch (error) {
    console.error('Failed to send message:', error);
    throw error;
  }
}
```

### Step 4: Monitor Session Events

Implement event monitoring to receive messages and updates:

#### TypeScript

```typescript
private async startEventMonitoring(): Promise<void> {
  if (!this.sessionId) return;

  while (true) {
    try {
      // Poll for new events with long polling
      const events = await this.client.sessions.listEvents(this.sessionId, {
        minOffset: this.lastOffset,
        waitForData: 30, // Wait up to 30 seconds for new events
        kinds: ["message", "status"] // Only get message and status events
      });

      // Process each event
      for (const event of events) {
        await this.handleEvent(event);
        this.lastOffset = Math.max(this.lastOffset, event.offset + 1);
      }

    } catch (error) {
      console.error('Event monitoring error:', error);
      // Wait before retrying
      await new Promise(resolve => setTimeout(resolve, 5000));
    }
  }
}

private async handleEvent(event: any): Promise<void> {
  if (event.kind === "message") {
    this.displayMessage(event);
  } else if (event.kind === "status") {
    this.updateStatus(event.data.status);
  }
}
```

#### JavaScript

```javascript
async startEventMonitoring() {
  if (!this.sessionId) return;

  while (true) {
    try {
      // Poll for new events with long polling
      const events = await this.client.sessions.listEvents(this.sessionId, {
        minOffset: this.lastOffset,
        waitForData: 30, // Wait up to 30 seconds for new events
        kinds: ["message", "status"] // Only get message and status events
      });

      // Process each event
      for (const event of events) {
        await this.handleEvent(event);
        this.lastOffset = Math.max(this.lastOffset, event.offset + 1);
      }

    } catch (error) {
      console.error('Event monitoring error:', error);
      // Wait before retrying
      await new Promise(resolve => setTimeout(resolve, 5000));
    }
  }
}

async handleEvent(event) {
  if (event.kind === "message") {
    this.displayMessage(event);
  } else if (event.kind === "status") {
    this.updateStatus(event.data.status);
  }
}
```

### Step 5: Display Messages in Your UI

Implement UI updates based on events from Parlant:

#### TypeScript

```typescript
private displayMessage(event: any): void {
  const messageElement = document.createElement('div');
  messageElement.className = `message ${event.source}`;

  // Style based on message source
  switch (event.source) {
    case 'customer':
      messageElement.classList.add('customer-message');
      break;
    case 'ai_agent':
      messageElement.classList.add('agent-message');
      break;
    case 'human_agent':
      messageElement.classList.add('human-agent-message');
      const agentName = event.data.participant?.display_name || 'Agent';
      messageElement.innerHTML = `
        <div class="agent-info">${agentName}</div>
        <div class="message-content">${event.data.message}</div>
      `;
      break;
  }

  // Add to chat container
  const chatContainer = document.getElementById('chat-messages');
  if (chatContainer) {
    chatContainer.appendChild(messageElement);
    chatContainer.scrollTop = chatContainer.scrollHeight;
  }
}

private updateStatus(status: string): void {
  const statusElement = document.getElementById('chat-status');
  if (statusElement) {
    switch (status) {
      case 'processing':
        statusElement.textContent = 'Agent is thinking...';
        break;
      case 'typing':
        statusElement.textContent = 'Agent is typing...';
        break;
      case 'ready':
        statusElement.textContent = '';
        break;
    }
  }
}
```

#### JavaScript

```javascript
displayMessage(event) {
  const messageElement = document.createElement('div');
  messageElement.className = `message ${event.source}`;

  // Style based on message source
  switch (event.source) {
    case 'customer':
      messageElement.classList.add('customer-message');
      messageElement.innerHTML = `<div class="message-content">${event.data.message}</div>`;
      break;
    case 'ai_agent':
      messageElement.classList.add('agent-message');
      messageElement.innerHTML = `<div class="message-content">${event.data.message}</div>`;
      break;
    case 'human_agent':
      messageElement.classList.add('human-agent-message');
      const agentName = event.data.participant?.display_name || 'Agent';
      messageElement.innerHTML = `
        <div class="agent-info">${agentName}</div>
        <div class="message-content">${event.data.message}</div>
      `;
      break;
  }

  // Add to chat container
  const chatContainer = document.getElementById('chat-messages');
  if (chatContainer) {
    chatContainer.appendChild(messageElement);
    chatContainer.scrollTop = chatContainer.scrollHeight;
  }
}

updateStatus(status) {
  const statusElement = document.getElementById('chat-status');
  if (statusElement) {
    switch (status) {
      case 'processing':
        statusElement.textContent = 'Agent is thinking...';
        break;
      case 'typing':
        statusElement.textContent = 'Agent is typing...';
        break;
      case 'ready':
        statusElement.textContent = '';
        break;
    }
  }
}
```

### Step 6: Complete HTML Example

Here's a complete HTML page that demonstrates the custom implementation:

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Custom Parlant Chat</title>
    <style>
        .chat-container {
            max-width: 500px;
            margin: 50px auto;
            border: 1px solid #ddd;
            border-radius: 8px;
            overflow: hidden;
        }

        .chat-header {
            background: #007bff;
            color: white;
            padding: 15px;
            text-align: center;
        }

        .chat-messages {
            height: 400px;
            padding: 15px;
            overflow-y: auto;
            background: #f8f9fa;
        }

        .message {
            margin: 10px 0;
            padding: 10px;
            border-radius: 8px;
            max-width: 80%;
        }

        .customer-message {
            background: #007bff;
            color: white;
            margin-left: auto;
            text-align: right;
        }

        .agent-message {
            background: white;
            border: 1px solid #ddd;
        }

        .human-agent-message {
            background: #28a745;
            color: white;
        }

        .chat-input {
            display: flex;
            padding: 15px;
            background: white;
        }

        .chat-input input {
            flex: 1;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 4px;
            margin-right: 10px;
        }

        .chat-input button {
            padding: 10px 20px;
            background: #007bff;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
        }

        #chat-status {
            font-style: italic;
            color: #666;
            padding: 5px 15px;
        }
    </style>
</head>
<body>
    <div class="chat-container">
        <div class="chat-header">
            <h3>Customer Support Chat</h3>
        </div>
        <div id="chat-status"></div>
        <div id="chat-messages" class="chat-messages"></div>
        <div class="chat-input">
            <input
                type="text"
                id="message-input"
                placeholder="Type your message..."
                onkeypress="handleKeyPress(event)"
            />
            <button onclick="sendMessage()">Send</button>
        </div>
    </div>

    <script type="module">
        import { ParlantClient } from 'https://unpkg.com/parlant-client@latest/dist/index.js';

        // Initialize your custom chat
        const chat = new ParlantChat('http://localhost:8800');

        // Start chat session
        chat.createSession('your-agent-id')
            .then(sessionId => {
                console.log('Chat ready!', sessionId);
            })
            .catch(error => {
                console.error('Failed to start chat:', error);
            });

        // Make functions available globally
        window.sendMessage = () => chat.sendUserMessage();
        window.handleKeyPress = (event) => {
            if (event.key === 'Enter') {
                chat.sendUserMessage();
            }
        };
    </script>
</body>
</html>
```

### Key Implementation Principles

1. **Event-Driven Architecture**: The chat is driven by events from Parlant sessions, ensuring consistency with the server state.
2. **Long Polling**: Use `waitForData` parameter in `listEvents()` for efficient real-time updates without constant polling.
3. **State Synchronization**: Always display what comes from Parlant events rather than optimistically updating the UI.
4. **Error Handling**: Implement robust error handling and retry logic for network issues.
5. **Responsive Design**: Ensure your chat interface works well on both desktop and mobile devices.

This approach gives you complete control over the chat experience while leveraging Parlant's powerful agent capabilities. You can adapt this pattern to any frontend framework or vanilla JavaScript implementation.
